import asyncio
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

MAIN_SOURCE = (BACKEND_DIR / "main.py").read_text(encoding="utf-8")

from services.personal_data_source_service import PersonalDataSourceService  # noqa: E402
from services.user_data_service import UserDataService  # noqa: E402


class FakeDataService:
    def get_available_years(self):
        return [27]

    def get_ls_range(self, _year):
        return (0.0, 360.0)

    def get_openmars_data(self, _year):
        return {"ls": [0.0, 360.0]}

    def get_aligned_mcd_data(self, _year):
        return {}

    def get_mcd_data(self, _year):
        return {}


class FastPersonalService(PersonalDataSourceService):
    def __init__(self):
        super().__init__(FakeDataService())
        self.scheduled = []
        self.inline_info_builds = 0
        self.inline_resolution_builds = 0

    async def _fetch_user_records(self, _user_id):
        return [object()]

    async def _get_build_state(self, _user_id):
        return None

    def _build_signature(self, _records):
        return ("sig",)

    def _schedule_build(self, user_id):
        self.scheduled.append(user_id)

    def _build_info_from_records_sync(self, *_args, **_kwargs):
        self.inline_info_builds += 1
        raise AssertionError("get_data_info should not build personal info inline")

    def _resolve_from_records_sync(self, *_args, **_kwargs):
        self.inline_resolution_builds += 1
        raise AssertionError("resolve_for_year should not resolve personal data inline")


def test_personal_data_info_returns_warming_state_without_inline_cache_build():
    async def run():
        service = FastPersonalService()

        info = await service.get_data_info("personal", user_id=7)

        assert service.scheduled == [7]
        assert service.inline_info_builds == 0
        assert info["source_meta"]["build_stage"] == "queued"
        assert info["source_meta"]["build_status"] == "building"

    asyncio.run(run())


def test_personal_year_resolution_returns_default_while_background_cache_builds():
    async def run():
        service = FastPersonalService()

        resolution = await service.resolve_for_year("personal", mars_year=27, user_id=7)

        assert service.scheduled == [7]
        assert service.inline_resolution_builds == 0
        assert resolution.effective_source == "default"
        assert resolution.build_stage == "queued"
        assert resolution.build_status == "building"

    asyncio.run(run())


def test_user_data_first_load_does_not_block_event_loop(tmp_path):
    async def run():
        service = UserDataService()
        upload_id = 123
        fake_file = tmp_path / "large.nc"
        fake_file.write_text("placeholder", encoding="utf-8")
        service._approved_index[upload_id] = fake_file

        def slow_load(_file_path):
            time.sleep(0.12)
            return {"lat": [], "lon": [], "ls": []}

        service._load_nc_file = slow_load

        load_task = asyncio.create_task(service._get_data(upload_id))
        tick_task = asyncio.create_task(asyncio.sleep(0.02))
        done, _pending = await asyncio.wait({load_task, tick_task}, timeout=0.06)

        assert tick_task in done
        await load_task

    asyncio.run(run())


def test_raw_upload_cache_rebuild_hook_is_noop():
    from types import SimpleNamespace
    from routers import upload as upload_router

    called = []

    class FakePersonalService:
        async def mark_build_queued(self, user_id):
            called.append(("mark", user_id))

        async def build_user_cache(self, user_id):
            called.append(("build", user_id))

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                enqueue_personal_cache_rebuild=lambda user_id: called.append(("enqueue", user_id)),
                personal_data_source_service=FakePersonalService(),
            )
        )
    )

    upload_router._enqueue_personal_cache_rebuild(request, 7)

    assert called == []


def test_mcd_cache_worker_runs_pending_jobs_from_lifespan():
    assert "from services.mcd_cache_service import McdCacheService" in MAIN_SOURCE
    assert "app.state.mcd_cache_service = mcd_cache_service" in MAIN_SOURCE
    assert "async def mcd_cache_worker()" in MAIN_SOURCE
    assert "await mcd_cache_service.run_next_pending_job()" in MAIN_SOURCE
    assert "app.state.enqueue_mcd_cache_job = enqueue_mcd_cache_job" in MAIN_SOURCE
    assert "UserDataService(mcd_cache_service=mcd_cache_service)" in MAIN_SOURCE
    assert "from services.official_mcd_source_interface import DisabledOfficialMcdSourcePublisher" in MAIN_SOURCE
    assert "app.state.official_mcd_source_publisher = DisabledOfficialMcdSourcePublisher()" in MAIN_SOURCE
