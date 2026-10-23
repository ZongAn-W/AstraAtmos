"""Response models for the public server dataset catalog.

Every nested structure is modeled explicitly so the public contract cannot
silently grow untyped fields. ``NaN``/``Infinity`` serialization is disabled.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Availability = Literal["available", "missing", "invalid", "unverified"]

MODEL_CONFIG = ConfigDict(allow_inf_nan=False)


class DatasetCapabilities(BaseModel):
    model_config = MODEL_CONFIG

    metadata: bool
    training: bool
    web_overview: bool
    trained_prediction: bool


class DatasetTime(BaseModel):
    model_config = MODEL_CONFIG

    kind: str
    calendar: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    count: Optional[int] = None
    step: Optional[int] = None
    step_unit: Optional[str] = None
    frequency_hours: Optional[int] = None
    time_zone: Optional[str] = None
    label: Optional[str] = None
    interval_start: Optional[str] = None
    interval_end_exclusive: Optional[str] = None


class DatasetGrid(BaseModel):
    model_config = MODEL_CONFIG

    shape: list[int] = Field(default_factory=list)
    dimension_order: list[str] = Field(default_factory=list)
    latitude_range: list[float] = Field(default_factory=list)
    longitude_range: list[float] = Field(default_factory=list)
    cell_bounds: Optional[dict[str, list[float]]] = None
    latitude_step: Optional[float] = None
    longitude_step: Optional[float] = None
    latitude_order: Optional[str] = None
    longitude_order: Optional[str] = None
    coverage: Optional[str] = None
    wrap_longitude: Optional[bool] = None
    latitude_values: list[float] = Field(default_factory=list)
    longitude_values: list[float] = Field(default_factory=list)


class DatasetVariable(BaseModel):
    model_config = MODEL_CONFIG

    id: str
    label: str
    units: str
    role: Literal["target_and_input", "optional_input"]
    missing_rate: Optional[float] = Field(default=None, ge=0, le=1)
    valid_mask: Optional[str] = None


class DatasetSplit(BaseModel):
    model_config = MODEL_CONFIG

    start: Optional[str]
    end: Optional[str]
    days: int
    steps: Optional[int] = None


class DatasetTrainingProfile(BaseModel):
    """Fixed training configuration published by the server for one dataset.

    Mars registrations keep this ``null``: their training entry predates the
    registry and is not described here. Earth publishes its profile so the
    training page renders real limits instead of guessing them, and so a client
    never has to hard-code the window, horizon, channels or units.
    """

    model_config = MODEL_CONFIG

    profile_id: str
    model_architectures: list[str] = Field(default_factory=list)
    model_sources: list[str] = Field(default_factory=list)
    target: str
    target_unit: str
    window: int
    horizon: int
    step_unit: str
    step: Optional[int] = None
    frequency_hours: Optional[int] = None
    optional_channels: list[str] = Field(default_factory=list)
    default_selected_channels: list[str] = Field(default_factory=list)
    input_units: dict[str, str] = Field(default_factory=dict)
    grid_shape: list[int] = Field(default_factory=list)
    supported_planet: str
    supports_sphere: bool
    supports_transfer_learning: bool
    implementation_id: Optional[str] = None


class DatasetDescriptor(BaseModel):
    # The public field name is ``schema``; the Python attribute avoids shadowing
    # the deprecated ``BaseModel.schema`` accessor.
    model_config = ConfigDict(allow_inf_nan=False, populate_by_name=True)

    dataset_id: str
    display_name: str
    planet: Literal["mars", "earth"]
    dataset_version: Optional[str] = None
    schema_: Optional[str] = Field(default=None, alias="schema")
    availability: Availability
    availability_reason: Optional[str] = None
    manifest_sha256: Optional[str] = None
    data_sha256: Optional[str] = None
    manifest_content_sha256: Optional[str] = None
    dataset_fingerprint: Optional[str] = None
    capabilities: DatasetCapabilities
    time: DatasetTime
    frequency_hours: Optional[int] = None
    step_unit: Optional[str] = None
    step: Optional[int] = None
    grid_shape: Optional[list[int]] = None
    grid: Optional[DatasetGrid] = None
    channel_order: list[str] = Field(default_factory=list)
    variables: list[DatasetVariable] = Field(default_factory=list)
    splits: Optional[dict[str, DatasetSplit]] = None
    training_profile: Optional[DatasetTrainingProfile] = None
    limitations: list[str] = Field(default_factory=list)


class DatasetListResponse(BaseModel):
    model_config = MODEL_CONFIG

    items: list[DatasetDescriptor]
    default_earth_dataset_id: str


__all__ = [
    "Availability",
    "DatasetCapabilities",
    "DatasetDescriptor",
    "DatasetGrid",
    "DatasetListResponse",
    "DatasetSplit",
    "DatasetTime",
    "DatasetTrainingProfile",
    "DatasetVariable",
]
