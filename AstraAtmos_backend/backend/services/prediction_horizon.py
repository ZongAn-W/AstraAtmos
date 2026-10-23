MAX_PREDICTION_HORIZON = 30


def _is_supported_integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def get_trained_horizon(hyperparameters: dict) -> int:
    horizon = hyperparameters.get("horizon") if isinstance(hyperparameters, dict) else None
    if (
        not _is_supported_integer(horizon)
        or horizon < 1
        or horizon > MAX_PREDICTION_HORIZON
    ):
        raise ValueError("Training task does not contain a valid trained output horizon")
    return horizon


def validate_prediction_horizon(hyperparameters: dict, requested_horizon: int) -> int:
    if (
        not _is_supported_integer(requested_horizon)
        or requested_horizon < 1
        or requested_horizon > MAX_PREDICTION_HORIZON
    ):
        raise ValueError(
            f"Prediction horizon must be between 1 and {MAX_PREDICTION_HORIZON}"
        )

    trained_horizon = get_trained_horizon(hyperparameters)
    if requested_horizon > trained_horizon:
        raise ValueError(
            f"requested prediction horizon {requested_horizon} "
            f"exceeds trained model horizon {trained_horizon}"
        )
    return requested_horizon
