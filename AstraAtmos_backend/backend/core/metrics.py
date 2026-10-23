"""
评估指标工具
专门用于计算 RMSE, MAE, R2, SSIM
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skimage.metrics import structural_similarity as ssim


MEAN_OVER_FORECAST_STEPS = "mean_over_forecast_steps"
POOLED_TEST_SET_PIXELS = "pooled_test_set_pixels"


def compute_metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    """
    计算逐步和总体评估指标。

    Args:
        truth: (horizon, H, W)
        pred:  (horizon, H, W)
    """
    horizon = truth.shape[0]
    per_step = []

    for step in range(horizon):
        t = truth[step].flatten()
        p = pred[step].flatten()

        valid = ~(np.isnan(t) | np.isnan(p))
        t_valid, p_valid = t[valid], p[valid]

        if len(t_valid) < 10:
            per_step.append(zero_metrics(step + 1))
            continue

        rmse = float(np.sqrt(mean_squared_error(t_valid, p_valid)))
        mae = float(mean_absolute_error(t_valid, p_valid))
        r2 = float(r2_score(t_valid, p_valid))

        t_2d = np.nan_to_num(truth[step], nan=0.0)
        p_2d = np.nan_to_num(pred[step], nan=0.0)
        data_range = max(t_2d.max() - t_2d.min(), 1e-10)
        ssim_val = float(ssim(t_2d, p_2d, data_range=data_range))

        per_step.append({
            "step": step + 1,
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "ssim": round(ssim_val, 4),
            "r2": round(r2, 4),
        })

    overall = {
        "step": 0,
        "rmse": round(float(np.mean([s["rmse"] for s in per_step])), 6),
        "mae": round(float(np.mean([s["mae"] for s in per_step])), 6),
        "ssim": round(float(np.mean([s["ssim"] for s in per_step])), 4),
        "r2": round(float(np.mean([s["r2"] for s in per_step])), 4),
    }

    return {
        "overall": overall,
        "per_step": per_step,
        "aggregation": {
            "overall": MEAN_OVER_FORECAST_STEPS,
            "per_step": "per_forecast_step",
        },
    }


def compute_test_set_metrics(truth: np.ndarray, pred: np.ndarray, horizon: int | None = None) -> dict:
    """
    Compute metrics over a test-set tensor.

    Args:
        truth: (N, horizon, H, W) or (N, horizon, 1, H, W)
        pred:  same shape as truth
    """
    truth_arr = _as_test_set_fields(truth)
    pred_arr = _as_test_set_fields(pred)
    actual_horizon = min(
        int(horizon or truth_arr.shape[1]),
        int(truth_arr.shape[1]),
        int(pred_arr.shape[1]),
    )
    if actual_horizon <= 0:
        return empty_metrics(0)

    per_step = []
    ssim_values = []
    for step in range(actual_horizon):
        t_step = truth_arr[:, step]
        p_step = pred_arr[:, step]
        t_valid, p_valid = _valid_flat_pair(t_step, p_step)
        if len(t_valid) < 10:
            per_step.append(zero_metrics(step + 1))
            continue

        rmse = float(np.sqrt(mean_squared_error(t_valid, p_valid)))
        mae = float(mean_absolute_error(t_valid, p_valid))
        try:
            r2 = float(r2_score(t_valid, p_valid))
        except Exception:
            r2 = 0.0

        step_ssim_values = []
        for sample_idx in range(t_step.shape[0]):
            t_2d = np.nan_to_num(t_step[sample_idx], nan=0.0)
            p_2d = np.nan_to_num(p_step[sample_idx], nan=0.0)
            data_range = max(float(t_2d.max() - t_2d.min()), 1e-10)
            try:
                step_ssim_values.append(float(ssim(t_2d, p_2d, data_range=data_range)))
            except Exception:
                pass
        ssim_val = float(np.mean(step_ssim_values)) if step_ssim_values else 0.0
        ssim_values.extend(step_ssim_values)

        per_step.append({
            "step": step + 1,
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "ssim": round(ssim_val, 4),
            "r2": round(r2, 4),
        })

    t_all, p_all = _valid_flat_pair(
        truth_arr[:, :actual_horizon],
        pred_arr[:, :actual_horizon],
    )
    if len(t_all) < 10:
        overall = zero_metrics(0)
    else:
        try:
            overall_r2 = float(r2_score(t_all, p_all))
        except Exception:
            overall_r2 = 0.0
        overall = {
            "step": 0,
            "rmse": round(float(np.sqrt(mean_squared_error(t_all, p_all))), 6),
            "mae": round(float(mean_absolute_error(t_all, p_all)), 6),
            "ssim": round(float(np.mean(ssim_values)) if ssim_values else 0.0, 4),
            "r2": round(overall_r2, 4),
        }

    return {
        "overall": overall,
        "per_step": per_step,
        "aggregation": {
            "overall": POOLED_TEST_SET_PIXELS,
            "per_step": "pooled_test_set_pixels_per_forecast_step",
        },
    }


def compute_error_distribution(
    truth: np.ndarray,
    pred: np.ndarray,
    bins: int = 50,
    max_points: int = 8000,
) -> dict:
    truth_arr = np.asarray(truth, dtype=np.float32)
    pred_arr = np.asarray(pred, dtype=np.float32)
    t_clean, p_clean = _valid_flat_pair(truth_arr, pred_arr)
    if len(t_clean) == 0:
        return empty_error_distribution()

    errors = p_clean - t_clean
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    sample_size = min(max(0, int(max_points)), len(t_clean))
    if sample_size > 0:
        indices = np.linspace(0, len(t_clean) - 1, sample_size, dtype=int)
        t_sample = t_clean[indices]
        p_sample = p_clean[indices]
    else:
        t_sample = np.array([])
        p_sample = np.array([])

    counts_t, edges_t = np.histogram(t_clean, bins=bins)
    counts_p, edges_p = np.histogram(p_clean, bins=bins)
    counts_e, edges_e = np.histogram(errors, bins=bins)

    return {
        "scatter": {
            "trues": t_sample.tolist(),
            "preds": p_sample.tolist(),
            "density": [1.0] * len(t_sample),
        },
        "hist_trues": {
            "bin_edges": edges_t.tolist(),
            "counts": counts_t.tolist(),
        },
        "hist_preds": {
            "bin_edges": edges_p.tolist(),
            "counts": counts_p.tolist(),
        },
        "hist_errors": {
            "bin_edges": edges_e.tolist(),
            "counts": counts_e.tolist(),
        },
        "mae": mae,
        "rmse": rmse,
    }


def empty_error_distribution() -> dict:
    return {
        "scatter": {"trues": [], "preds": [], "density": []},
        "hist_trues": {"bin_edges": [], "counts": []},
        "hist_preds": {"bin_edges": [], "counts": []},
        "hist_errors": {"bin_edges": [], "counts": []},
        "mae": 0.0,
        "rmse": 0.0,
    }


def _as_test_set_fields(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 5:
        arr = arr[:, :, 0]
    if arr.ndim != 4:
        raise ValueError(f"Expected test-set fields with 4 or 5 dimensions, got {arr.ndim}")
    return arr


def _valid_flat_pair(truth: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(truth).reshape(-1)
    p = np.asarray(pred).reshape(-1)
    valid = ~(np.isnan(t) | np.isnan(p))
    return t[valid], p[valid]


def zero_metrics(step: int) -> dict:
    return {"step": step, "rmse": 0.0, "mae": 0.0, "ssim": 0.0, "r2": 0.0}


def empty_metrics(horizon: int) -> dict:
    per_step = [
        {"step": i + 1, "rmse": 0.0, "mae": 0.0, "ssim": 0.0, "r2": 0.0}
        for i in range(horizon)
    ]
    return {
        "overall": {"step": 0, "rmse": 0.0, "mae": 0.0, "ssim": 0.0, "r2": 0.0},
        "per_step": per_step,
        "aggregation": {
            "overall": POOLED_TEST_SET_PIXELS,
            "per_step": "pooled_test_set_pixels_per_forecast_step",
        },
    }
