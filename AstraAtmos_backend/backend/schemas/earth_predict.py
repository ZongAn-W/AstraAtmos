"""Earth historical prediction response models.

The public contract keeps daily dates and adds explicit UTC timestamps for the
three-hour release, along with the real grid, DU fields and ordered lead metrics.
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Optional

from pydantic import BaseModel, ConfigDict, Field, model_serializer

MODEL_CONFIG = ConfigDict(allow_inf_nan=False)


class _ConditionalTemporalFields(BaseModel):
    """Omit absent new fields while retaining every legacy default/null field.

    Global exclude_unset/exclude_none would also alter old grid/model responses.
    Only the additive temporal fields need conditional serialization.
    """

    conditional_fields: ClassVar[frozenset[str]] = frozenset()

    @model_serializer(mode="wrap")
    def _omit_absent_temporal_fields(self, handler):
        result = handler(self)
        for name in self.conditional_fields:
            if name not in self.model_fields_set:
                result.pop(name, None)
                alias = type(self).model_fields[name].serialization_alias
                if alias:
                    result.pop(alias, None)
        return result


class EarthPredictGrid(BaseModel):
    model_config = MODEL_CONFIG

    shape: list[int] = Field(default_factory=list)
    latitude: list[float] = Field(default_factory=list)
    longitude: list[float] = Field(default_factory=list)
    latitude_range: Optional[list[float]] = None
    longitude_range: Optional[list[float]] = None
    latitude_step: Optional[float] = None
    longitude_step: Optional[float] = None
    coverage: Optional[str] = None
    wrap_longitude: Optional[bool] = None


class EarthPredictVariable(BaseModel):
    model_config = MODEL_CONFIG

    id: str
    label: str
    units: str
    role: str


class EarthPredictFieldDay(BaseModel):
    model_config = MODEL_CONFIG

    field: list[list[float]]
    minVal: float
    maxVal: float
    valid_cells: int


class EarthLeadMetric(_ConditionalTemporalFields):
    model_config = MODEL_CONFIG

    conditional_fields = frozenset({"lead_day", "lead_step", "lead_hours", "mse", "r2", "mape", "smape", "target_statistics"})

    lead_day: Optional[int] = Field(default=None, ge=1)
    lead_step: Optional[int] = Field(default=None, ge=1)
    lead_hours: Optional[int] = Field(default=None, ge=1)
    rmse: float
    mae: float
    mse: Optional[float] = None
    r2: Optional[float] = None
    mape: Optional[float] = None
    smape: Optional[float] = None
    target_statistics: Optional[dict[str, float]] = None


class EarthHorizonMetric(_ConditionalTemporalFields):
    model_config = MODEL_CONFIG
    conditional_fields = frozenset({"mse", "r2", "mape", "smape"})

    horizon_hours: int = Field(..., ge=1)
    lead_steps: int = Field(..., ge=1)
    rmse: float
    mae: float
    mse: Optional[float] = None
    r2: Optional[float] = None
    mape: Optional[float] = None
    smape: Optional[float] = None


class EarthPredictMetrics(_ConditionalTemporalFields):
    model_config = MODEL_CONFIG

    conditional_fields = frozenset({"by_horizon", "metrics_schema", "metric_units", "metric_policy"})

    unit: str
    target: str
    aggregation: str
    reference_available: bool
    overall: dict[str, float] = Field(default_factory=dict)
    by_lead: list[EarthLeadMetric] = Field(default_factory=list)
    by_horizon: Optional[list[EarthHorizonMetric]] = None
    metrics_schema: Optional[str] = Field(default=None, alias="schema")
    metric_units: Optional[dict[str, str]] = None
    metric_policy: Optional[dict[str, Any]] = None


class EarthPredictOrigins(_ConditionalTemporalFields):
    model_config = MODEL_CONFIG

    conditional_fields = frozenset({
        "kind", "time_zone", "frequency_hours", "step_unit", "step", "timestamps",
        "input_offset_days", "target_offset_days", "input_offset_hours", "target_offset_hours",
    })

    start: str
    end: str
    count: int
    dates: list[str] = Field(default_factory=list)
    window: int
    horizon: int
    input_offset_days: Optional[int] = None
    target_offset_days: Optional[int] = None
    kind: Optional[str] = None
    time_zone: Optional[str] = None
    frequency_hours: Optional[int] = None
    step_unit: Optional[str] = None
    step: Optional[int] = None
    timestamps: Optional[list[str]] = None
    input_offset_hours: Optional[int] = None
    target_offset_hours: Optional[int] = None


class EarthPredictRunBlock(BaseModel):
    model_config = MODEL_CONFIG

    best_epoch: Optional[int] = None
    epochs_completed: Optional[int] = None
    seed: Optional[int] = None
    device: Optional[str] = None
    created_utc: Optional[str] = None
    linear_hidden_layers: Optional[int] = None


class EarthPredictModelIdentity(BaseModel):
    """Which model produced (or will produce) the prediction.

    Published so the page can name the trained model and, for an uploaded one, show
    its pinned package, version and content hash - never a server file path.
    """

    model_config = MODEL_CONFIG

    model_source: Optional[str] = None
    model_architecture: Optional[str] = None
    uploaded_model_id: Optional[str] = None
    uploaded_model_name: Optional[str] = None
    uploaded_model_version: Optional[int] = None
    uploaded_model_content_hash: Optional[str] = None
    uploaded_model_source_embedded: Optional[bool] = None


class EarthPredictContextResponse(_ConditionalTemporalFields):
    model_config = MODEL_CONFIG

    conditional_fields = frozenset({"frequency_hours", "step_unit", "step", "time_zone", "timestamp_rule"})

    planet: str = "earth"
    task_id: int
    dataset_id: str
    dataset_version: str
    dataset_fingerprint: str
    target: str
    target_unit: str
    window: int
    horizon: int
    frequency_hours: Optional[int] = None
    step_unit: Optional[str] = None
    step: Optional[int] = None
    time_zone: Optional[str] = None
    timestamp_rule: Optional[str] = None
    input_channel_order: list[str] = Field(default_factory=list)
    input_units: list[str] = Field(default_factory=list)
    grid: EarthPredictGrid
    origins: EarthPredictOrigins
    training_split_end: Optional[str] = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    run: EarthPredictRunBlock = Field(default_factory=EarthPredictRunBlock)
    model: EarthPredictModelIdentity = Field(default_factory=EarthPredictModelIdentity)
    warnings: list[str] = Field(default_factory=list)


class EarthPredictRequest(BaseModel):
    model_config = MODEL_CONFIG

    training_task_id: int = Field(..., ge=1)
    forecast_origin: str = Field(..., min_length=8, max_length=40)


class EarthCompareRequest(BaseModel):
    model_config = MODEL_CONFIG

    task_ids: list[Annotated[int, Field(strict=True, ge=1)]] = Field(..., min_length=2, max_length=32)




class EarthPredictResponse(_ConditionalTemporalFields):
    export_ref: Optional[dict[str, Any]] = None
    model_config = MODEL_CONFIG

    conditional_fields = frozenset({
        "input_timestamps", "target_timestamps", "frequency_hours", "step_unit", "step",
        "time_zone", "timestamp_rule", "cache_key",
    })

    planet: str = "earth"
    task_id: int
    dataset_id: str
    dataset_version: str
    dataset_fingerprint: str
    target: str
    target_unit: str
    model_architecture: Optional[str] = None
    model_source: Optional[str] = None
    model: EarthPredictModelIdentity = Field(default_factory=EarthPredictModelIdentity)
    warnings: list[str] = Field(default_factory=list)
    input_channel_order: list[str] = Field(default_factory=list)
    input_units: list[str] = Field(default_factory=list)
    forecast_origin: str
    origin_split: str
    input_dates: list[str] = Field(default_factory=list)
    target_dates: list[str] = Field(default_factory=list)
    input_timestamps: Optional[list[str]] = None
    cache_key: Optional[str] = None
    target_timestamps: Optional[list[str]] = None
    frequency_hours: Optional[int] = None
    step_unit: Optional[str] = None
    step: Optional[int] = None
    time_zone: Optional[str] = None
    timestamp_rule: Optional[str] = None
    window: int
    horizon: int
    grid: EarthPredictGrid
    variables: list[EarthPredictVariable] = Field(default_factory=list)
    prediction: list[EarthPredictFieldDay] = Field(default_factory=list)
    reference: list[EarthPredictFieldDay] = Field(default_factory=list)
    residual: list[EarthPredictFieldDay] = Field(default_factory=list)
    metrics: EarthPredictMetrics


__all__ = [
    "EarthCompareRequest",
    "EarthLeadMetric",
    "EarthHorizonMetric",
    "EarthPredictContextResponse",
    "EarthPredictFieldDay",
    "EarthPredictGrid",
    "EarthPredictMetrics",
    "EarthPredictModelIdentity",
    "EarthPredictOrigins",
    "EarthPredictRequest",
    "EarthPredictResponse",
    "EarthPredictRunBlock",
    "EarthPredictVariable",
]
