import numpy as np
import pandas as pd
import unittest

from four_g_protocol import (
    CELL_ID_COL,
    DATE_COL,
    FEATURE_COLS,
    build_windows,
    build_window_refs,
    fit_training_scaler,
    load_observations,
)
from inference_engine import _infer_statistical


def _frame(rows):
    data = []
    for cell_id, dates in rows:
        for index, date in enumerate(dates):
            data.append({
                DATE_COL: date,
                CELL_ID_COL: cell_id,
                **{column: float(index + offset) for offset, column in enumerate(FEATURE_COLS)},
            })
    return pd.DataFrame(data)


class FourGProtocolTests(unittest.TestCase):
    def test_windows_never_cross_cells_or_time_gaps(self):
        dates_a = pd.date_range("2025-01-01", periods=6, freq="h")
        dates_b = pd.date_range("2025-01-01", periods=5, freq="h").delete(3)
        frame = _frame([("a", dates_a), ("b", dates_b)])
        scaler = fit_training_scaler(frame)

        x, y, refs = build_windows(frame, scaler, seq_len=2, pred_len=2, step=1)

        self.assertEqual(x.shape, (3, 2, 8))
        self.assertEqual(y.shape, (3, 2, 8))
        self.assertEqual([ref.cell_id for ref in refs], ["a", "a", "a"])

    def test_window_refs_are_stable_when_rows_are_interleaved(self):
        dates = pd.date_range("2025-01-01", periods=4, freq="h")
        frame = _frame([("b", dates), ("a", dates)]).sample(frac=1, random_state=7)

        refs = build_window_refs(frame, seq_len=2, pred_len=2, step=1)

        self.assertEqual([(ref.cell_id, ref.start) for ref in refs], [
            ("a", pd.Timestamp("2025-01-01")),
            ("b", pd.Timestamp("2025-01-01")),
        ])

    def test_base_file_is_rejected_as_observations(self):
        with self.assertRaisesRegex(ValueError, "external forecasts"):
            load_observations("base")

    def test_autoregression_is_not_the_linear_trend_baseline(self):
        series = np.array([0.0, 1.0] * 12, dtype=np.float64).reshape(1, 24, 1)
        autoar = _infer_statistical("autoar", series, pred_len=6)
        linear = _infer_statistical("linear_regression", series, pred_len=6)
        self.assertFalse(np.allclose(autoar, linear))
