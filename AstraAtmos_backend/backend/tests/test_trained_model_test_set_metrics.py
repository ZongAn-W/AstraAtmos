import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.metrics import compute_error_distribution, compute_metrics, compute_test_set_metrics  # noqa: E402


def test_test_set_metrics_aggregate_all_samples_and_steps():
    truth = np.arange(2 * 2 * 8 * 8, dtype=np.float32).reshape(2, 2, 8, 8)
    pred = truth.copy()
    pred[:, 0] += 1.0
    pred[:, 1] += 3.0

    metrics = compute_test_set_metrics(truth, pred, horizon=2)

    assert metrics["per_step"][0]["rmse"] == 1.0
    assert metrics["per_step"][0]["mae"] == 1.0
    assert metrics["per_step"][1]["rmse"] == 3.0
    assert metrics["per_step"][1]["mae"] == 3.0
    assert metrics["overall"]["rmse"] == round(float(np.sqrt(5.0)), 6)
    assert metrics["overall"]["mae"] == 2.0
    assert metrics["aggregation"] == {
        "overall": "pooled_test_set_pixels",
        "per_step": "pooled_test_set_pixels_per_forecast_step",
    }


def test_single_prediction_metrics_identify_step_mean_aggregation():
    truth = np.arange(2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8)
    pred = truth.copy()

    metrics = compute_metrics(truth, pred)

    assert metrics["aggregation"] == {
        "overall": "mean_over_forecast_steps",
        "per_step": "per_forecast_step",
    }


def test_error_distribution_uses_all_values_for_histograms_and_metrics():
    truth = np.arange(2 * 2 * 8 * 8, dtype=np.float32).reshape(2, 2, 8, 8)
    pred = truth + 2.0

    distribution = compute_error_distribution(truth, pred, bins=8, max_points=12)

    assert len(distribution["scatter"]["trues"]) == 12
    assert sum(distribution["hist_trues"]["counts"]) == truth.size
    assert sum(distribution["hist_preds"]["counts"]) == pred.size
    assert sum(distribution["hist_errors"]["counts"]) == truth.size
    assert distribution["mae"] == 2.0
    assert distribution["rmse"] == 2.0


if __name__ == "__main__":
    test_test_set_metrics_aggregate_all_samples_and_steps()
    test_error_distribution_uses_all_values_for_histograms_and_metrics()
    print("trained model test-set metric tests passed")
