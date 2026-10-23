"""
Schemas for exploration/overview endpoints.
"""

from pydantic import BaseModel


class SourceMeta(BaseModel):
    requested_source: str
    effective_source: str
    fallback: bool = False
    message: str | None = None
    mars_year: int | None = None
    build_status: str | None = None
    build_stage: str | None = None
    build_progress: float | None = None
    build_stage_message: str | None = None
    signature_hash: str | None = None
    upload_id: int | None = None
    upload_filename: str | None = None
    data_type: str | None = None


class GlobePoint(BaseModel):
    lat: float
    lng: float
    val: float


class ValidationPoint(GlobePoint):
    mcd_value: float
    nomad_value: float
    count: int


class GlobeDataResponse(BaseModel):
    points: list[GlobePoint]
    minVal: float
    maxVal: float
    ls: float
    mars_year: int
    variable: str = "o3col"
    source_meta: SourceMeta | None = None


class PointProbeRequested(BaseModel):
    lat: float
    lng: float
    ls: float


class PointProbeGridPoint(BaseModel):
    lat: float
    lng: float


class PointProbeCurrent(BaseModel):
    ls: float
    value: float | None = None


class PointProbeSeries(BaseModel):
    ls: list[float]
    point: list[float | None]
    globalMean: list[float | None]
    latitudeMean: list[float | None]


class PointProbeComparison(BaseModel):
    globalMean: float | None = None
    latitudeMean: float | None = None
    pointMinusGlobal: float | None = None
    pointMinusLatitudeMean: float | None = None


class PointProbeResponse(BaseModel):
    requested: PointProbeRequested
    gridPoint: PointProbeGridPoint
    current: PointProbeCurrent
    series: PointProbeSeries
    comparison: PointProbeComparison
    variable: str = "o3col"
    mars_year: int
    source_meta: SourceMeta | None = None


class HeatmapResponse(BaseModel):
    x: list[float]
    y: list[float]
    z: list[list[float | None]]
    min: float
    max: float
    variable: str = "o3col"
    source_meta: SourceMeta | None = None


class LatitudeBand(BaseModel):
    name: str
    values: list[float]


class SeasonalBandsResponse(BaseModel):
    ls: list[float]
    bands: list[LatitudeBand]
    source_meta: SourceMeta | None = None


class CorrelationResponse(BaseModel):
    matrix: list[list[float | None]]
    variable_names: list[str]
    source_meta: SourceMeta | None = None


class OverviewTimeline(BaseModel):
    min: float
    max: float
    step: float


class OverviewInfoResponse(BaseModel):
    available_years: list[int]
    timeline: OverviewTimeline
    ozone_capabilities: dict
    source_meta: SourceMeta | None = None


class OzoneSourceLayerResponse(BaseModel):
    source: str
    points: list[GlobePoint]
    minVal: float
    maxVal: float
    ls: float


class NomadValidationResponse(BaseModel):
    source: str
    comparison: str
    matched_ls: float
    sample_count: int
    bias: float
    mae: float
    rmse: float
    correlation: float | None = None
    minDiff: float
    maxDiff: float
    points: list[ValidationPoint]


class OzoneValidationResponse(BaseModel):
    nomad: NomadValidationResponse | None = None


class OverviewOzoneSourcesResponse(BaseModel):
    mars_year: int
    requested_ls: float
    anchor_ls: float
    mcd: OzoneSourceLayerResponse
    openmars: OzoneSourceLayerResponse | None = None
    nomad: OzoneSourceLayerResponse | None = None
    available_sources: list[str]
    diff_candidates: list[str]
    validation: OzoneValidationResponse | None = None
    capabilities: dict
