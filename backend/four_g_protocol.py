"""Trusted data contract for the 4G benchmark.

The parquet files are a panel: each cell ID is an independent hourly series.
Windows must therefore be built inside one cell and one uninterrupted time span.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from .project_paths import DATA_DIR
except ImportError:  # Supports direct execution from backend/ during development.
    from project_paths import DATA_DIR

DATE_COL = "date"
CELL_ID_COL = "ID编号"
FEATURE_COLS = (
    "erab流量", "pdcch利用率", "pdsch利用率", "pusch利用率",
    "上行流量", "下行流量", "总流量", "有效连接数",
)
STATIC_COLS = (CELL_ID_COL, "厂商", "频段", "场景")
SEQ_LEN = 24
PRED_LEN = 24
STEP = 3


@dataclass(frozen=True)
class WindowRef:
    cell_id: str
    start: pd.Timestamp
    segment_start: int


def dataset_path(split: str) -> Path:
    files = {
        "train": "df_4g_train_100.parquet",
        "test": "df_4g_test_100.parquet",
        "base": "df_4g_base_100.parquet",
    }
    if split not in files:
        raise ValueError(f"Unknown 4G dataset split: {split}")
    return DATA_DIR / files[split]


def load_observations(split: str) -> pd.DataFrame:
    """Load observed traffic, never the external base-model forecast file."""
    if split == "base":
        raise ValueError("The base file contains external forecasts and cannot be observations.")
    path = dataset_path(split)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    validate_observations(frame, source=str(path))
    return frame


def validate_observations(frame: pd.DataFrame, source: str = "data") -> None:
    required = {DATE_COL, CELL_ID_COL, *FEATURE_COLS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    if frame[DATE_COL].isna().any() or frame[CELL_ID_COL].isna().any():
        raise ValueError(f"{source} contains missing date or cell ID values")
    if frame[list(FEATURE_COLS)].isna().any().any():
        raise ValueError(f"{source} contains missing 4G feature values")


def split_continuous_segments(frame: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield hourly-continuous segments separately for each cell.

    Sorting here makes the contract independent of parquet row order. Gaps split a
    segment rather than creating a fabricated sequence through the missing hours.
    """
    validate_observations(frame)
    ordered = frame.sort_values([CELL_ID_COL, DATE_COL], kind="stable")
    for cell_id, group in ordered.groupby(CELL_ID_COL, sort=False):
        group = group.reset_index(drop=True)
        dates = pd.to_datetime(group[DATE_COL])
        duplicate = dates.duplicated(keep=False)
        if duplicate.any():
            duplicate_values = dates[duplicate].astype(str).head(3).tolist()
            raise ValueError(f"Cell {cell_id} has duplicate timestamps: {duplicate_values}")
        boundaries = dates.diff().ne(pd.Timedelta(hours=1)).to_numpy()
        boundaries[0] = True
        segment_ids = boundaries.cumsum()
        for _, segment in group.groupby(segment_ids, sort=False):
            yield str(cell_id), segment.reset_index(drop=True)


def build_window_refs(
    frame: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    pred_len: int = PRED_LEN,
    step: int = STEP,
) -> list[WindowRef]:
    if min(seq_len, pred_len, step) <= 0:
        raise ValueError("seq_len, pred_len, and step must be positive")
    span = seq_len + pred_len
    refs: list[WindowRef] = []
    for cell_id, segment in split_continuous_segments(frame):
        for start in range(0, len(segment) - span + 1, step):
            refs.append(WindowRef(cell_id, pd.Timestamp(segment.at[start, DATE_COL]), start))
    return refs


def build_windows(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    seq_len: int = SEQ_LEN,
    pred_len: int = PRED_LEN,
    step: int = STEP,
) -> tuple[np.ndarray, np.ndarray, list[WindowRef]]:
    """Create scaled (X, Y) windows without crossing a cell or time gap."""
    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    refs: list[WindowRef] = []
    span = seq_len + pred_len
    for cell_id, segment in split_continuous_segments(frame):
        values = scaler.transform(segment.loc[:, FEATURE_COLS].to_numpy(dtype=np.float64))
        for start in range(0, len(segment) - span + 1, step):
            rows_x.append(values[start:start + seq_len])
            rows_y.append(values[start + seq_len:start + span])
            refs.append(WindowRef(cell_id, pd.Timestamp(segment.at[start, DATE_COL]), start))
    shape_x = (0, seq_len, len(FEATURE_COLS))
    shape_y = (0, pred_len, len(FEATURE_COLS))
    return (
        np.asarray(rows_x, dtype=np.float32).reshape(shape_x if not rows_x else (-1, seq_len, len(FEATURE_COLS))),
        np.asarray(rows_y, dtype=np.float32).reshape(shape_y if not rows_y else (-1, pred_len, len(FEATURE_COLS))),
        refs,
    )


def temporal_train_validation_split(frame: pd.DataFrame, validation_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each cell chronologically, keeping validation strictly later than training."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    for _, segment in split_continuous_segments(frame):
        cutoff = int(len(segment) * (1 - validation_fraction))
        train_parts.append(segment.iloc[:cutoff])
        validation_parts.append(segment.iloc[cutoff:])
    return pd.concat(train_parts, ignore_index=True), pd.concat(validation_parts, ignore_index=True)


def fit_training_scaler(train_frame: pd.DataFrame) -> StandardScaler:
    validate_observations(train_frame, source="training observations")
    return StandardScaler().fit(train_frame.loc[:, FEATURE_COLS].to_numpy(dtype=np.float64))
