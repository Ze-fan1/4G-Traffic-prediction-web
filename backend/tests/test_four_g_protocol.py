import numpy as np
import pandas as pd
import unittest
import json
import tempfile
from pathlib import Path

from four_g_protocol import (
    CELL_ID_COL,
    DATE_COL,
    FEATURE_COLS,
    build_windows,
    build_window_refs,
    fit_training_scaler,
    load_observations,
)
from inference_engine import _infer_statistical, _forecast_autoar, _forecast_autoarima
from generic_forecast import forecast
from benchmark_artifacts import write_benchmark_artifact
from data_pipeline import build_single_series_window
from four_g_protocol import WindowRef
from generic_forecast import forecast


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

    def test_autoar_and_autoarima_stay_finite_on_an_unstable_short_series(self):
        series = np.array([0.0, 5.0, -4.0, 8.0, -7.0, 12.0] * 4)
        for forecast in (_forecast_autoar(series, 24), _forecast_autoarima(series, 24)):
            self.assertEqual(forecast.shape, (24,))
            self.assertTrue(np.isfinite(forecast).all())
            self.assertLessEqual(np.max(np.abs(forecast - series.mean())), max(4 * series.std(), 2 * np.ptp(series)))

    def test_single_series_adapter_does_not_duplicate_channels(self):
        frame = pd.DataFrame({"custom_metric": np.arange(30, dtype=np.float32)})
        windows, scaler = build_single_series_window(frame, "custom_metric")

        self.assertEqual(windows.shape, (8, 24, 8))
        for channel in range(8):
            self.assertGreater(float(np.std(windows[channel, :, channel])), 0)
            self.assertTrue(np.allclose(np.delete(windows[channel], channel, axis=1), 0))
        self.assertEqual(scaler.n_features_in_, 1)

    def test_generic_forecast_backtests_and_predicts_in_original_scale(self):
        values = np.sin(np.arange(60) / 3) + 10
        result = forecast(pd.DataFrame({"custom_metric": values}), "custom_metric", 6, "autoar")
        self.assertEqual(result.prediction.shape, (6,))
        self.assertGreaterEqual(result.validation_mae, 0)

    def test_artifact_records_window_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            refs = [WindowRef("cell-a", pd.Timestamp("2025-01-01"), 0)]
            manifest = write_benchmark_artifact(
                directory, "Demo", np.zeros((1, 2, 8)), np.ones((1, 2, 8)), refs, "test"
            )
            saved = json.loads((Path(directory) / "benchmark_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["protocol"], "4g-panel-v1")
            self.assertEqual(manifest["window_refs"][0]["cell_id"], "cell-a")

    def test_generic_forecast_backtests_and_predicts_in_original_scale(self):
        values = np.sin(np.arange(60) / 3) + 10
        result = forecast(pd.DataFrame({"custom_metric": values}), "custom_metric", 6, "autoar")
        self.assertEqual(result.prediction.shape, (6,))
        self.assertGreaterEqual(result.validation_mae, 0)
