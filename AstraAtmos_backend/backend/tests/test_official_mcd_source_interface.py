import asyncio
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.official_mcd_source_interface import (  # noqa: E402
    DisabledOfficialMcdSourcePublisher,
    OfficialMcdPromotionResult,
)


def test_disabled_official_publisher_returns_disabled_result_without_side_effects(tmp_path):
    async def run():
        publisher = DisabledOfficialMcdSourcePublisher()
        result = await publisher.publish_upload_to_official(
            upload_id=42,
            reviewer_id=1,
            reason="approved by admin",
        )

        assert isinstance(result, OfficialMcdPromotionResult)
        assert result.status == "disabled"
        assert result.upload_id == 42
        assert result.mars_year is None
        assert "not enabled" in result.message
        assert list(tmp_path.iterdir()) == []

    asyncio.run(run())
