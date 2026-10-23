"""Bounded task-test diagnostics; clients cannot supply data identities."""
from pydantic import BaseModel, ConfigDict, Field


class EarthDiagnosticsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    training_task_id: int = Field(..., strict=True, ge=1)
    sample_windows: int = Field(default=4, strict=True, ge=1, le=8)
    scatter_points: int = Field(default=5000, strict=True, ge=1, le=20000)
    histogram_bins: int = Field(default=40, strict=True, ge=5, le=100)
    seed: int = Field(default=42, strict=True, ge=0, le=2**32 - 1)
    pfi_repeats: int = Field(default=3, strict=True, ge=1, le=5)
    include_pfi: bool = Field(default=True, strict=True)
