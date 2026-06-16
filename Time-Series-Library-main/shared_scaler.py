"""
Shared StandardScaler for consistent model comparison.
======================================================
All models must use the SAME StandardScaler and column ordering
to produce comparable predictions in the same coordinate space.

This module provides the canonical scaler that matches
- Dataset_Custom.__read_data__() in data_provider/data_loader.py
- generate_web_data.py

Column order (scaler index):
  0: ERAB流量
  1: PDCCH利用率
  2: PDSCH利用率
  3: PUSCH利用率
  4: 上行流量
  5: 下行流量
  6: 有效连接数
  7: 总流量  ← target, moved to LAST

Usage:
  from shared_scaler import get_shared_scaler, get_feature_order, prepare_dataframe
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_PKL_PATH = os.path.join(os.path.dirname(__file__), 'shared_scaler.pkl')
_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_provider', '4g_traffic')
_TARGET = '总流量'
_DROP_COLS = ['ID编号', '厂商', '频段', '场景']


def get_feature_order():
    """Return the canonical 8-channel column order (target last).

    Returns:
        list[str]: Column names in order [ERAB, PDCCH, PDSCH, PUSCH,
                  上行, 下行, 有效连接, 总流量]
    """
    train_fp = os.path.join(_DATA_DIR, 'df_4g_train_100.parquet')
    df = pd.read_parquet(train_fp)
    for col in _DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
    cols = list(df.columns)
    cols.remove(_TARGET)
    cols.remove('date')
    return cols + [_TARGET]


def prepare_dataframe(df):
    """Reorder DataFrame columns to match the canonical order.

    Drops static/non-numeric columns, moves '总流量' to the last position,
    and returns a DataFrame with columns: [date, feat0, feat1, ..., 总流量]

    Args:
        df: Raw DataFrame from parquet

    Returns:
        DataFrame with canonical column order (including 'date' as first column)
    """
    df = df.copy()
    for col in _DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
    cols = list(df.columns)
    cols.remove(_TARGET)
    if 'date' in cols:
        cols.remove('date')
    return df[['date'] + cols + [_TARGET]]


def get_shared_scaler(force_recreate=False):
    """Return the shared StandardScaler, creating/caching it as needed.

    The scaler is fit on the FULL training set (df_4g_train_100.parquet)
    with canonical column ordering, matching generate_web_data.py exactly.

    Args:
        force_recreate: If True, delete cached scaler and re-fit

    Returns:
        sklearn.preprocessing.StandardScaler
    """
    if os.path.exists(_PKL_PATH) and not force_recreate:
        with open(_PKL_PATH, 'rb') as f:
            return pickle.load(f)

    train_fp = os.path.join(_DATA_DIR, 'df_4g_train_100.parquet')
    df_train = pd.read_parquet(train_fp)
    df_train = prepare_dataframe(df_train)

    # Drop 'date' column for scaler fitting
    train_data = df_train[df_train.columns[1:]].values

    scaler = StandardScaler().fit(train_data)

    with open(_PKL_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    print(f'[shared_scaler] Scaler fitted on {train_data.shape[0]} samples × {train_data.shape[1]} channels')
    print(f'[shared_scaler] Feature order: {list(df_train.columns[1:])}')
    print(f'[shared_scaler] Saved to {_PKL_PATH}')
    return scaler


def scale_dataframe(df, scaler=None):
    """Scale a prepared DataFrame using the shared scaler.

    Args:
        df: DataFrame already prepared by prepare_dataframe()
        scaler: Optional scaler (loads shared if not provided)

    Returns:
        np.ndarray of shape (n_samples, 8) in scaled space
    """
    if scaler is None:
        scaler = get_shared_scaler()
    return scaler.transform(df[df.columns[1:]].values)


# ── Module-level cache ──
if __name__ == '__main__':
    scaler = get_shared_scaler(force_recreate=True)
    print(f'Mean: {scaler.mean_}')
    print(f'Scale: {scaler.scale_}')
    print(f'Feature order: {get_feature_order()}')
