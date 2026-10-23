from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class OfficialMcdPromotionResult:
    upload_id: int
    status: str
    message: str
    mars_year: int | None = None


class OfficialMcdSourcePublisher(Protocol):
    async def publish_upload_to_official(
        self,
        upload_id: int,
        reviewer_id: int,
        reason: str = "",
    ) -> OfficialMcdPromotionResult:
        ...


class DisabledOfficialMcdSourcePublisher:
    async def publish_upload_to_official(
        self,
        upload_id: int,
        reviewer_id: int,
        reason: str = "",
    ) -> OfficialMcdPromotionResult:
        return OfficialMcdPromotionResult(
            upload_id=int(upload_id),
            status="disabled",
            message="Official MCD source publishing is not enabled in this phase.",
            mars_year=None,
        )
