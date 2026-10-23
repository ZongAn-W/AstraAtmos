from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class UserModelEarthVerdict(BaseModel):
    """Whether the uploaded model may also be trained on Earth data.

    Published on every upload so the training page can explain - before a task is
    created - that a model valid for Mars is not automatically usable on Earth.
    """

    compatible: bool = False
    errors: List[str] = Field(default_factory=list)
    output_shape: Optional[List[int]] = None


class UserModelEarthDatasetVerdict(UserModelEarthVerdict):
    dataset_id: str
    status: str = "unknown"
    code: Optional[str] = None
    contract_schema: Optional[str] = None


class UserModelValidationReport(BaseModel):
    ok: bool = False
    code: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    output_shape: Optional[List[int]] = None
    #: Declared dataset feeds (currently only ``earth_merra2``); omitted for models
    #: that do not opt in, which keeps the historical report shape.
    datasets: Optional[Dict[str, Any]] = None
    earth: Optional[UserModelEarthVerdict] = None
    earth_datasets: Optional[Dict[str, UserModelEarthDatasetVerdict]] = None
    mars: Optional[Dict[str, bool]] = None


class UserModelPackageResponse(BaseModel):
    id: str
    user_id: int
    display_name: str
    version: int
    original_filename: str
    content_hash: str
    param_schema: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None
    validation_status: str
    validation_report: UserModelValidationReport
    created_at: Any
    updated_at: Any


class UserModelListResponse(BaseModel):
    items: List[UserModelPackageResponse]


class UserModelRenameRequest(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=120, strict=True)
