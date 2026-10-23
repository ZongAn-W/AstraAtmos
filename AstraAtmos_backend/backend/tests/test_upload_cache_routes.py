import asyncio
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from database.models import UploadRecord  # noqa: E402
from routers import upload as upload_router  # noqa: E402
from services.upload_service import ValidationResult  # noqa: E402


def test_upload_success_payload_contains_cache_building_state():
    async def run():
        result = upload_router._success_payload(
            UploadRecord(
                id=42,
                user_id=7,
                filename="MCD_MY33_global_3h_5deg_10m_ls_lst.nc",
                file_path="/tmp/original.nc",
                data_type="mcd",
                mars_year=33,
                status="cache_building",
            ),
            ValidationResult(
                is_valid=True,
                data_type="mcd",
                mars_year=33,
                ls_start=0.0,
                ls_end=360.0,
                lat_points=36,
                lon_points=72,
                variables=["O3COL"],
            ),
            cache_status={"jobs": [{"job_type": "mcd_overview", "status": "pending", "progress": 0.0}]},
        )

        assert result["status"] == "cache_building"
        assert result["cache_status"]["jobs"][0]["job_type"] == "mcd_overview"

    asyncio.run(run())


def test_upload_list_item_includes_cache_status_for_mcd():
    item = upload_router._upload_list_item(
        UploadRecord(
            id=42,
            user_id=7,
            filename="MCD_MY33_global_3h_5deg_10m_ls_lst.nc",
            file_path="/tmp/original.nc",
            file_size=10,
            data_type="mcd",
            mars_year=33,
            ls_start=0.0,
            ls_end=360.0,
            status="cache_building",
            is_public=False,
        ),
        cache_status={
            "jobs": [{"job_type": "mcd_overview", "status": "running", "progress": 55.0}],
            "artifacts": [],
        },
    )

    assert item["cache_status"]["jobs"][0]["progress"] == 55.0
    assert item["status"] == "cache_building"
