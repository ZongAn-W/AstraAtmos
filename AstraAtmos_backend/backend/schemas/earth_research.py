"""Request/response models for the Earth scientific analysis endpoints.

Every response model carries ``allow_inf_nan=False``: a ``NaN`` or ``Infinity``
that escaped the numeric layer becomes a serialization error instead of a
silently ``null`` entry in a numeric array. Nested containers are declared
explicitly, so a field cannot be dropped by an untyped ``dict``.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, FiniteFloat, model_serializer

MODEL_CONFIG = ConfigDict(allow_inf_nan=False, populate_by_name=True)
REQUEST_CONFIG = ConfigDict(allow_inf_nan=False, extra="forbid", populate_by_name=True)

VariableId = Literal["TO3", "U10M", "V10M", "T2M", "SWGDN"]
ScopeId = Literal["global", "south_polar", "south_mid", "tropics", "north_mid", "north_polar"]
BandId = Literal["south_polar", "south_mid", "tropics", "north_mid", "north_polar"]
Hemisphere = Literal["south", "north"]
FiniteNumber = Annotated[FiniteFloat, Field()]
Fingerprint = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
OptionalFinite = Optional[FiniteNumber]
OptionalReason = Optional[str]


# ── shared building blocks ─────────────────────────────────────────────

class _AdditiveEarthFields(BaseModel):
    conditional_fields: ClassVar[frozenset[str]] = frozenset({
        "frequency_hours", "time_zone", "time_aggregation", "analysis_aggregation", "timestamp_rule",
        "source_grid_shape", "render_grid_shape", "render_method",
    })

    @model_serializer(mode="wrap")
    def _omit_absent_additions(self, handler):
        result = handler(self)
        for key in self.conditional_fields:
            if key not in self.model_fields_set:
                result.pop(key, None)
        return result


class EarthResearchIdentity(_AdditiveEarthFields):
    model_config = MODEL_CONFIG

    planet: Literal["earth"]
    dataset_id: str
    dataset_version: Optional[str] = None
    dataset_fingerprint: str
    frequency_hours: Optional[int] = None
    time_zone: Optional[str] = None
    time_aggregation: Optional[str] = None
    dataset_schema: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("dataset_schema", "schema"),
        serialization_alias="schema",
    )


class EarthSourceMeta(_AdditiveEarthFields):
    model_config = MODEL_CONFIG

    source: str
    cadence: str
    dataset_schema: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("dataset_schema", "schema"),
        serialization_alias="schema",
    )
    dataset_version: Optional[str] = None
    source_files_count: Optional[int] = None
    source_sha256: Optional[str] = None
    processing: Optional[str] = None
    calendar: str
    frequency_hours: Optional[int] = None
    time_zone: Optional[str] = None
    analysis_aggregation: Optional[str] = None


class EarthTimeModel(_AdditiveEarthFields):
    model_config = MODEL_CONFIG

    kind: Literal["iso-date", "iso-datetime"]
    calendar: str
    start: str
    end: str
    count: int
    step: int
    step_unit: Literal["day", "hour"]
    years: list[int]
    frequency_hours: Optional[int] = None
    time_zone: Optional[str] = None
    timestamp_rule: Optional[str] = None


class EarthColorRange(BaseModel):
    model_config = MODEL_CONFIG

    min: FiniteNumber
    max: FiniteNumber
    scope: Literal["dataset"]
    centered_on_zero: bool


class EarthAnomalyColorRange(BaseModel):
    model_config = MODEL_CONFIG

    min: FiniteNumber
    max: FiniteNumber
    centered_on_zero: Literal[True]


class EarthVariable(BaseModel):
    model_config = MODEL_CONFIG

    id: VariableId
    label: str
    units: str
    color_range: EarthColorRange


class EarthVariableBrief(BaseModel):
    model_config = MODEL_CONFIG

    id: VariableId
    units: str


class EarthGeometry(BaseModel):
    model_config = MODEL_CONFIG

    shape: list[int]
    dimension_order: list[Literal["lat", "lon"]]
    lat_centers: list[FiniteNumber]
    lon_centers: list[FiniteNumber]
    lat_bounds: list[FiniteNumber]
    lon_bounds: list[FiniteNumber]
    cell_bounds: dict[str, list[FiniteNumber]]
    latitude_step: FiniteNumber
    longitude_step: FiniteNumber
    wrap_longitude: bool
    coverage: str


class EarthCapabilities(BaseModel):
    model_config = MODEL_CONFIG

    field: bool
    playback: bool
    pointProbe: bool
    polar: bool
    diurnal: bool
    researchSuite: bool
    spatialDiagnostics: bool
    aiInsight: bool


class EarthBand(BaseModel):
    model_config = MODEL_CONFIG

    id: BandId
    min_latitude: OptionalFinite = None
    max_latitude: OptionalFinite = None
    latitude_values: list[FiniteNumber]
    grid_point_count: int


class EarthPolarBand(BaseModel):
    model_config = MODEL_CONFIG

    id: BandId
    hemisphere: Optional[Hemisphere] = None
    min_latitude: OptionalFinite = None
    max_latitude: OptionalFinite = None
    latitude_values: list[FiniteNumber]
    grid_point_count: int
    label: Optional[str] = None


class EarthPolarScope(BaseModel):
    model_config = MODEL_CONFIG

    min_abs_latitude: FiniteNumber
    bands: list[EarthPolarBand]
    sampling: Literal["daily_mean"]
    note: str


class EarthContextResponse(EarthResearchIdentity):
    source_meta: EarthSourceMeta
    time: EarthTimeModel
    variables: list[EarthVariable]
    geometry: EarthGeometry
    capabilities: EarthCapabilities
    unavailable: dict[str, str]
    polar_scope: EarthPolarScope
    limitations: list[str]


# ── research suite ─────────────────────────────────────────────────────

class EarthSeasonalSeries(BaseModel):
    model_config = MODEL_CONFIG

    # z[i][t]: latitude rows in ascending order, then the daily series.
    z: list[list[FiniteNumber]]
    units: str
    aggregation: Literal["equal_longitude_mean"]


class EarthExtremes(BaseModel):
    model_config = MODEL_CONFIG

    max_value: OptionalFinite = None
    max_date: Optional[str] = None
    min_value: OptionalFinite = None
    min_date: Optional[str] = None
    peak_to_peak: OptionalFinite = None


class EarthZScoreSeries(BaseModel):
    model_config = MODEL_CONFIG

    values: list[OptionalFinite]
    reason: Optional[str] = None


class EarthBandSeries(BaseModel):
    """Daily band series per variable; ``None`` means the band has no rows."""

    model_config = MODEL_CONFIG

    TO3: Optional[list[FiniteNumber]] = None
    U10M: Optional[list[FiniteNumber]] = None
    V10M: Optional[list[FiniteNumber]] = None
    T2M: Optional[list[FiniteNumber]] = None
    SWGDN: Optional[list[FiniteNumber]] = None


class EarthBandExtremes(BaseModel):
    model_config = MODEL_CONFIG

    TO3: EarthExtremes
    U10M: EarthExtremes
    V10M: EarthExtremes
    T2M: EarthExtremes
    SWGDN: EarthExtremes


class EarthRegionalSeries(BaseModel):
    model_config = MODEL_CONFIG

    TO3: list[FiniteNumber]
    U10M: list[FiniteNumber]
    V10M: list[FiniteNumber]
    T2M: list[FiniteNumber]
    SWGDN: list[FiniteNumber]


class EarthSeasonalBlock(BaseModel):
    model_config = MODEL_CONFIG

    TO3: EarthSeasonalSeries
    U10M: EarthSeasonalSeries
    V10M: EarthSeasonalSeries
    T2M: EarthSeasonalSeries
    SWGDN: EarthSeasonalSeries


class EarthZScoreBlock(BaseModel):
    model_config = MODEL_CONFIG

    TO3: EarthZScoreSeries
    U10M: EarthZScoreSeries
    V10M: EarthZScoreSeries
    T2M: EarthZScoreSeries
    SWGDN: EarthZScoreSeries


class EarthBandSeriesBlock(BaseModel):
    model_config = MODEL_CONFIG

    south_polar: EarthBandSeries
    south_mid: EarthBandSeries
    tropics: EarthBandSeries
    north_mid: EarthBandSeries
    north_polar: EarthBandSeries


class EarthBandExtremesBlock(BaseModel):
    model_config = MODEL_CONFIG

    south_polar: EarthBandExtremes
    south_mid: EarthBandExtremes
    tropics: EarthBandExtremes
    north_mid: EarthBandExtremes
    north_polar: EarthBandExtremes


class EarthAggregationModel(BaseModel):
    model_config = MODEL_CONFIG

    regional: Literal["spherical_cell_area_mean"]
    seasonal: Literal["equal_longitude_mean"]
    band: Literal["cell_area_weighted_band_mean"]
    scope: Literal["global"]


class EarthRelationshipScope(BaseModel):
    model_config = MODEL_CONFIG

    kind: Literal["global", "band"]
    band_id: Optional[BandId] = None
    units_note: str


class EarthCorrelationPair(BaseModel):
    model_config = MODEL_CONFIG

    r: OptionalFinite = None
    n: int
    reason: OptionalReason = None


class EarthCorrelationMatrix(BaseModel):
    model_config = MODEL_CONFIG

    variable_order: list[VariableId]
    r: list[list[OptionalFinite]]
    n: list[list[int]]
    reason: list[list[OptionalReason]]
    pairs: dict[str, EarthCorrelationPair]


class EarthLagPoint(BaseModel):
    model_config = MODEL_CONFIG

    lag_days: int
    r: OptionalFinite = None
    n: int
    reason: OptionalReason = None


class EarthRegression(BaseModel):
    model_config = MODEL_CONFIG

    slope: FiniteNumber
    intercept: FiniteNumber


class EarthDriverRelationship(BaseModel):
    """``x = driver``, ``y = TO3``; ``slope_units`` names the slope unit."""

    model_config = MODEL_CONFIG

    driver: VariableId
    target: Literal["TO3"]
    slope_units: str
    r: OptionalFinite = None
    n: int
    reason: OptionalReason = None
    regression: Optional[EarthRegression] = None
    lag: list[EarthLagPoint]


class EarthRelationships(BaseModel):
    model_config = MODEL_CONFIG

    scope: EarthRelationshipScope
    correlation: EarthCorrelationMatrix
    solar_ozone: EarthDriverRelationship
    temperature_ozone: EarthDriverRelationship


class EarthResearchSuiteResponse(EarthResearchIdentity):
    year: int
    start: str
    end: str
    day_count: int
    dates: list[str]
    latitude: list[FiniteNumber]
    bands: list[EarthBand]
    variables: list[EarthVariableBrief]
    aggregation: EarthAggregationModel
    seasonal: EarthSeasonalBlock
    regional_series: EarthRegionalSeries
    band_series: EarthBandSeriesBlock
    extremes: EarthBandExtremesBlock
    zscore: EarthZScoreBlock
    relationships: dict[str, EarthRelationships]


# ── spatial diagnostics ────────────────────────────────────────────────

class EarthSpatialBand(BaseModel):
    model_config = MODEL_CONFIG

    id: BandId
    rms: OptionalFinite = None
    peak_to_peak: OptionalFinite = None
    grid_point_count: int


class EarthSpatialDiagnosticsResponse(EarthResearchIdentity):
    source_grid_shape: Optional[list[int]] = None
    render_grid_shape: Optional[list[int]] = None
    render_method: Optional[str] = None
    year: int
    variable: VariableId
    units: str
    lat: list[FiniteNumber]
    lon: list[FiniteNumber]
    anomaly: list[list[FiniteNumber]]
    reference: Literal["annual_mean_minus_equal_longitude_mean"]
    color_range: EarthAnomalyColorRange
    bands: list[EarthSpatialBand]


# ── polar dynamics ─────────────────────────────────────────────────────

class EarthPolarVariable(BaseModel):
    model_config = MODEL_CONFIG

    units: str
    mean: OptionalFinite = None
    min_value: OptionalFinite = None
    min_date: Optional[str] = None
    max_value: OptionalFinite = None
    max_date: Optional[str] = None
    peak_to_peak: OptionalFinite = None
    series: list[FiniteNumber]
    reason: OptionalReason = None
    days: int


class EarthPolarBandVariables(BaseModel):
    model_config = MODEL_CONFIG

    TO3: EarthPolarVariable
    U10M: EarthPolarVariable
    V10M: EarthPolarVariable
    T2M: EarthPolarVariable
    SWGDN: EarthPolarVariable


class EarthPolarDynamicsBand(BaseModel):
    model_config = MODEL_CONFIG

    id: BandId
    hemisphere: Optional[Hemisphere] = None
    min_latitude: OptionalFinite = None
    max_latitude: OptionalFinite = None
    latitude_values: list[FiniteNumber]
    grid_point_count: int
    days: int
    variables: EarthPolarBandVariables


class EarthPolarScopeBrief(BaseModel):
    model_config = MODEL_CONFIG

    min_abs_latitude: FiniteNumber
    sampling: Literal["daily_mean"]
    note: str


class EarthHemisphereContrast(BaseModel):
    model_config = MODEL_CONFIG

    n: int
    r: OptionalFinite = None
    reason: OptionalReason = None
    mean_difference: OptionalFinite = None


class EarthHemisphereContrastBlock(BaseModel):
    model_config = MODEL_CONFIG

    TO3: EarthHemisphereContrast
    U10M: EarthHemisphereContrast
    V10M: EarthHemisphereContrast
    T2M: EarthHemisphereContrast
    SWGDN: EarthHemisphereContrast


class EarthPolarDynamicsResponse(EarthResearchIdentity):
    year: int
    start: str
    end: str
    day_count: int
    dates: list[str]
    polar_scope: EarthPolarScopeBrief
    bands: list[EarthPolarDynamicsBand]
    hemisphere_contrast: EarthHemisphereContrastBlock


# ── AI insight ─────────────────────────────────────────────────────────

class EarthInsightCard(BaseModel):
    """One supplementary client card: numbers stay server-side, notes are text."""

    model_config = REQUEST_CONFIG

    card: str = Field(max_length=64)
    values: dict[str, object] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list, max_length=50)


class EarthInsightSummary(BaseModel):
    """Statistics digest only.

    A raw ``[36][72]`` field must never be sent here: the endpoint recomputes
    every number from the verified release, and a request carrying a full grid
    is rejected as ``insight_payload_too_large``.
    """

    model_config = REQUEST_CONFIG

    units: Optional[str] = None
    cards: list[EarthInsightCard] = Field(default_factory=list)


class EarthInsightRequest(BaseModel):
    model_config = REQUEST_CONFIG

    planet: str = Field(max_length=32)
    dataset_id: str = Field(max_length=128)
    expected_fingerprint: Optional[Fingerprint] = None
    year: int
    variable: VariableId
    date: Optional[str] = Field(default=None, max_length=32)
    scope: ScopeId = "global"
    question: Optional[str] = Field(default=None, max_length=2000)
    locale: Literal["zh", "en"] = "zh"
    summary: Optional[EarthInsightSummary] = None


class EarthInsightDigestCard(BaseModel):
    """One digest card. Values are small scalars or short labels, never grids."""

    model_config = MODEL_CONFIG

    card: str
    values: dict[str, Optional[FiniteFloat | str]]
    notes: list[str]


class EarthInsightDigestRange(BaseModel):
    model_config = MODEL_CONFIG

    start: str
    end: str


class EarthInsightDigest(BaseModel):
    model_config = MODEL_CONFIG

    planet: Literal["earth"]
    dataset_id: str
    dataset_version: Optional[str] = None
    dataset_fingerprint: str
    source: str
    cadence: str
    calendar: str
    year: int
    date_range: EarthInsightDigestRange
    variable: VariableId
    units: str
    scope: ScopeId
    scope_label: str
    day_count: int
    date: str
    requested_date: Optional[str] = None
    engine: str
    cards: list[EarthInsightDigestCard]
    units_note: str


class EarthInsightResponse(EarthResearchIdentity):
    source: str
    year: int
    date_range: EarthInsightDigestRange
    variable: VariableId
    units: str
    scope: ScopeId
    scope_label: str
    time_kind: Literal["iso-date"]
    calendar: str
    model: str
    answer: str
    digest: EarthInsightDigest
    limitations: list[str]
    generated_at: str


__all__ = [
    "EarthBand",
    "EarthBandExtremes",
    "EarthBandExtremesBlock",
    "EarthBandSeries",
    "EarthBandSeriesBlock",
    "EarthCapabilities",
    "EarthColorRange",
    "EarthContextResponse",
    "EarthCorrelationMatrix",
    "EarthCorrelationPair",
    "EarthDriverRelationship",
    "EarthExtremes",
    "EarthGeometry",
    "EarthHemisphereContrast",
    "EarthHemisphereContrastBlock",
    "EarthInsightCard",
    "EarthInsightDigest",
    "EarthInsightDigestCard",
    "EarthInsightRequest",
    "EarthInsightResponse",
    "EarthInsightSummary",
    "EarthLagPoint",
    "EarthPolarBand",
    "EarthPolarBandVariables",
    "EarthPolarDynamicsBand",
    "EarthPolarDynamicsResponse",
    "EarthPolarScope",
    "EarthPolarScopeBrief",
    "EarthPolarVariable",
    "EarthRegionalSeries",
    "EarthRegression",
    "EarthRelationships",
    "EarthResearchIdentity",
    "EarthResearchSuiteResponse",
    "EarthSeasonalBlock",
    "EarthSeasonalSeries",
    "EarthSourceMeta",
    "EarthSpatialBand",
    "EarthSpatialDiagnosticsResponse",
    "EarthTimeModel",
    "EarthVariable",
    "EarthVariableBrief",
    "EarthZScoreBlock",
    "EarthZScoreSeries",
]
