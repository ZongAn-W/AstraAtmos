"""Strict, bounded options for server-rendered scientific figures."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExportSource(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str = Field(min_length=32, max_length=36)
    planet: Literal["mars", "earth"]
    analysis: Literal["prediction", "metrics", "pfi"]
    task_id: int = Field(ge=1)
    horizon: int = Field(ge=1, le=30)
    origin: str = Field(max_length=40)
    variables: list[str] = Field(default_factory=list, max_length=6)


class FigureOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    width_mm: float = Field(default=180, ge=70, le=240)
    height_mm: float = Field(default=85, ge=55, le=260)
    language: Literal["zh", "en"] = "en"
    font: Literal["auto", "sans", "serif"] = "auto"
    font_size: float = Field(default=10, ge=6, le=16)
    line_width: float = Field(default=0.8, ge=0.3, le=3)
    colormap: Literal["inferno", "viridis", "plasma", "magma", "cividis", "jet", "rdbu"] = "viridis"
    format: Literal["pdf", "svg", "png"] = "pdf"
    dpi: Literal[300, 600] = 300
    panel_labels: bool = True
    include_title: bool = True
    unit: Literal["um-atm", "DU"] = "um-atm"

    @model_validator(mode="after")
    def bounded_pixels(self):
        if self.width_mm * self.height_mm * (self.dpi / 25.4) ** 2 > 24_000_000:
            raise ValueError("Figure exceeds 24 megapixels; reduce size or DPI")
        return self


class ResearchExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    sources: list[ExportSource] = Field(min_length=1, max_length=8)
    kind: Literal["triptych", "scatter", "step_curves", "pfi"]
    step: int = Field(default=0, ge=0, le=29)
    metric: Literal["rmse", "mae", "r2", "ssim"] = "rmse"
    options: FigureOptions = Field(default_factory=FigureOptions)
    bundle: bool = False
