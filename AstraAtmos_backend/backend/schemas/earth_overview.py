"""Response models for the read-only 2D Earth overview endpoints.

``NaN``/``Infinity`` serialization is disabled. Three-hour missing observations
are explicitly mapped from the validated validity mask to ``null``; unexpected
non-finite numeric values remain schema errors.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_serializer

MODEL_CONFIG = ConfigDict(allow_inf_nan=False)

VariableId = Literal["TO3", "U10M", "V10M", "T2M", "SWGDN"]
FiniteNumber = Annotated[FiniteFloat, Field()]


class EarthResponseIdentity(BaseModel):
    model_config = MODEL_CONFIG

    dataset_id: str
    dataset_version: Optional[str] = None
    dataset_fingerprint: str
    planet: Literal["earth"]
    variable: VariableId
    units: str
    frequency_hours: Optional[int] = None
    step_unit: Optional[Literal["hour", "day"]] = None
    step: Optional[int] = None
    time_zone: Optional[Literal["UTC"]] = None
    source_grid_shape: Optional[list[int]] = None
    missing_policy: Optional[str] = None

    @model_serializer(mode="wrap")
    def omit_absent_additions(self, handler):
        result = handler(self)
        for key in ("frequency_hours", "step_unit", "step", "time_zone", "source_grid_shape",
                    "missing_policy", "timestamp", "timestamps", "grid_shape", "render_grid_shape", "render", "render_method"):
            if key not in self.model_fields_set:
                result.pop(key, None)
        return result


class EarthCoverage(BaseModel):
    model_config = MODEL_CONFIG

    latitude_range: list[FiniteNumber]
    longitude_range: list[FiniteNumber]
    wrap_longitude: bool


class EarthColorRange(BaseModel):
    model_config = MODEL_CONFIG

    min: FiniteNumber
    max: FiniteNumber
    scope: Literal["dataset"]
    centered_on_zero: bool


class EarthFieldStatistics(BaseModel):
    model_config = MODEL_CONFIG

    min: Optional[FiniteNumber]
    max: Optional[FiniteNumber]
    regional_mean: Optional[FiniteNumber]
    valid_count: int


class EarthFieldResponse(EarthResponseIdentity):
    date: str
    timestamp: Optional[str] = None
    grid_shape: Optional[list[int]] = None
    render_grid_shape: Optional[list[int]] = None
    render_method: Optional[str] = None
    render: Optional[dict] = None
    calendar: str
    lat: list[FiniteNumber]
    lon: list[FiniteNumber]
    dimension_order: list[Literal["lat", "lon"]]
    field: list[list[Optional[FiniteNumber]]]
    coverage: EarthCoverage
    color_range: EarthColorRange
    statistics: EarthFieldStatistics


class EarthRegionalSeriesResponse(EarthResponseIdentity):
    start: str
    end: str
    dates: list[str]
    timestamps: Optional[list[str]] = None
    values: list[Optional[FiniteNumber]]
    aggregation: Literal["cos_lat_sample_mean", "spherical_cell_area_mean"]
    coverage: EarthCoverage


class EarthGridPoint(BaseModel):
    model_config = MODEL_CONFIG

    lat: FiniteNumber
    lon: FiniteNumber
    lat_index: int
    lon_index: int


class EarthRequestedPoint(BaseModel):
    model_config = MODEL_CONFIG

    lat: FiniteNumber
    lon: FiniteNumber


class EarthPointSeriesResponse(EarthResponseIdentity):
    start: str
    end: str
    requested: EarthRequestedPoint
    grid_point: EarthGridPoint
    selection: Literal["nearest_grid_point"]
    dates: list[str]
    timestamps: Optional[list[str]] = None
    values: list[Optional[FiniteNumber]]


__all__ = [
    "EarthColorRange",
    "EarthCoverage",
    "EarthFieldResponse",
    "EarthFieldStatistics",
    "EarthGridPoint",
    "EarthPointSeriesResponse",
    "EarthRegionalSeriesResponse",
    "EarthRequestedPoint",
    "EarthResponseIdentity",
]
