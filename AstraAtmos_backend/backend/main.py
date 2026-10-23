"""
AstraAtmos 后端入口
- 启动时预加载数据和模型（lifespan）
- 注册所有 API 路由
- 配置 CORS（允许前端跨域）
"""

import logging
import os
import time
import sys
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager, suppress

# Windows 下异步子进程必须使用 ProactorEventLoop
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, FileResponse

from config import (
    API_PREFIX,
    DEFAULT_EARTH_DATASET_ID,
    EARTH_MERRA2_DIR,
    EARTH_MERRA2_V1_DIR,
    EARTH_MERRA2_3HOURLY_DIR,
    EARTH_MERRA2_3HOURLY_5DEG_DIR,
    EARTH_TRAINING_CACHE_DIR,
    OVERVIEW_MCD_VARIABLES,
    PENDING_REVIEW_DIR,
    TRAINING_WEIGHTS_DIR,
    USER_MODELS_DIR,
    USER_UPLOADS_DIR,
)
from database.init_db import init_database
from database.engine import async_session_maker
from services.data_service import DataService
from services.analysis_service import AnalysisService
from services.mcd_overview_data_service import McdOverviewDataService
from services.inference_service import InferenceService
from services.ai_service import AIService
from services.copilot_service import CopilotService
from services.upload_service import UploadService
from services.mcd_cache_service import McdCacheService
from services.official_mcd_source_interface import DisabledOfficialMcdSourcePublisher
from services.user_model_service import UserModelService
from services.training_weight_service import TrainingWeightService
from services.user_data_service import UserDataService
from services.personal_data_source_service import PersonalDataSourceService
from services.data_governance_service import DataGovernanceService
from services.dataset_registry import DatasetRegistry
from services.earth_overview_service import EarthOverviewService
from services.earth_research_service import EarthResearchService
from services.personal_data_source_service import SingleYearDataView
from core.analysis_transforms import AnalysisTransforms
from routers import analysis, predict, ai, copilot
from routers import auth
from routers import upload as upload_router_module
from routers import governance as governance_router_module
from routers import notification as notification_router_module
from routers import user_data as user_data_router_module
from routers import feedback as feedback_router_module
from routers import training as training_router_module
from routers import user_models as user_models_router_module
from routers import datasets as datasets_router_module
from routers import earth_overview as earth_overview_router_module
from routers import earth_analysis as earth_analysis_router_module
from routers import earth_predict as earth_predict_router_module
from routers import research_export as research_export_router_module

# ─── 日志配置 ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("astraatmos")


def _resolve_frontend_dist_dir() -> Path | None:
    """
    Resolve frontend dist directory for single-process deployment.
    Priority:
    1) ASTRAATMOS_FRONTEND_DIST env
    2) <backend>/frontend_dist           (portable package layout)
    3) <repo>/frontend/dist              (local dev/release build)
    """
    candidates: list[Path] = []
    env_dir = os.getenv("ASTRAATMOS_FRONTEND_DIST", "").strip()
    if env_dir:
        candidates.append(Path(env_dir))

    backend_dir = Path(__file__).resolve().parent
    repo_root = backend_dir.parent.parent
    candidates.append(backend_dir / "frontend_dist")
    candidates.append(repo_root / "frontend" / "dist")

    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if (resolved / "index.html").is_file():
            logger.info(f"检测到前端静态资源目录: {resolved}")
            return resolved
    logger.info("未检测到前端静态资源目录，后端以 API-only 模式运行")
    return None


FRONTEND_DIST_DIR = _resolve_frontend_dist_dir()


# ─── 生命周期：启动时预加载，关闭时清理 ───

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan 事件：
    - 启动时：读取 nc 数据到内存 + 加载 PyTorch 模型
    - 关闭时：释放资源
    """
    logger.info("=" * 60)
    logger.info("  AstraAtmos 后端启动中 (已重构架构)...")
    logger.info("=" * 60)

    t0 = time.time()

    # 0. 数据库初始化（建表 + 默认管理员账号）
    logger.info("[0/4] 初始化数据库...")
    await init_database()
    app.state.db_session = async_session_maker
    await training_router_module.training_service.start()

    # 确保上传目录存在
    USER_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    PENDING_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    USER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 基础服务：数据加载
    logger.info("[1/4] 初始化基础数据加载服务...")
    data_service = DataService()
    app.state.data_service = data_service

    mcd_overview_service = McdOverviewDataService(data_service)
    app.state.mcd_overview_service = mcd_overview_service

    # 上传服务（依赖 data_service）
    upload_service = UploadService(data_service)
    app.state.upload_service = upload_service

    mcd_cache_service = McdCacheService()
    app.state.mcd_cache_service = mcd_cache_service
    app.state.official_mcd_source_publisher = DisabledOfficialMcdSourcePublisher()
    mcd_cache_event = asyncio.Event()

    def enqueue_mcd_cache_job() -> None:
        mcd_cache_event.set()

    async def mcd_cache_worker() -> None:
        while True:
            await mcd_cache_event.wait()
            mcd_cache_event.clear()
            while await mcd_cache_service.run_next_pending_job():
                await asyncio.sleep(0)

    app.state.enqueue_mcd_cache_job = enqueue_mcd_cache_job
    app.state.mcd_cache_worker_task = asyncio.create_task(mcd_cache_worker())
    enqueue_mcd_cache_job()

    app.state.user_model_service = UserModelService(storage_root=USER_MODELS_DIR)
    app.state.training_weight_service = TrainingWeightService(storage_root=TRAINING_WEIGHTS_DIR)

    # 用户数据服务（按需读取用户上传的 .nc 文件）
    user_data_service = UserDataService(mcd_cache_service=mcd_cache_service)
    app.state.user_data_service = user_data_service

    # 数据源解析服务（默认/个人数据源切换 + 自动降级）
    personal_source_service = PersonalDataSourceService(data_service)
    app.state.personal_data_source_service = personal_source_service
    personal_cache_rebuild_queue: asyncio.Queue[int] = asyncio.Queue()
    personal_cache_pending_users: set[int] = set()

    def enqueue_personal_cache_rebuild(user_id: int) -> None:
        try:
            uid = int(user_id)
        except (TypeError, ValueError):
            return
        if uid <= 0:
            return
        if uid in personal_cache_pending_users:
            return
        personal_cache_pending_users.add(uid)
        personal_cache_rebuild_queue.put_nowait(uid)

    async def warm_personal_runtime_caches(user_id: int) -> None:
        status = await personal_source_service.get_build_status(user_id)
        if status.get("status") != "ready":
            return

        info = await personal_source_service.get_data_info("personal", user_id)
        years = list(info.get("available_years") or [])
        if not years:
            return

        primary_year = int(years[0])

        await personal_source_service._upsert_build_state(
            user_id=user_id,
            signature_hash=status.get("signature_hash") or "",
            status="building",
            stage="warming_analysis",
            progress=72.0,
            stage_message=personal_source_service._build_stage_message("warming_analysis"),
            error=None,
            duration_ms=status.get("duration_ms"),
            built_at=None,
        )

        resolution = await personal_source_service.resolve_for_year("personal", primary_year, user_id)
        if resolution.effective_source != "default":
            analysis_cache = getattr(app.state, "personal_analysis_service_cache", None)
            if analysis_cache is None:
                from cachetools import LRUCache
                analysis_cache = LRUCache(maxsize=16)
                app.state.personal_analysis_service_cache = analysis_cache

            analysis_key = (
                int(user_id),
                int(resolution.mars_year),
                str(resolution.effective_source),
                str(getattr(resolution, "signature_hash", "") or ""),
            )
            analysis_service_cached = analysis_cache.get(analysis_key)
            if analysis_service_cached is None:
                data_view = SingleYearDataView(
                    mars_year=resolution.mars_year,
                    openmars_data=resolution.openmars_data,
                    aligned_mcd_data=resolution.aligned_mcd_data,
                    mcd_raw_data=resolution.mcd_raw_data,
                )
                analysis_service_cached = AnalysisService(data_view)
                analysis_cache[analysis_key] = analysis_service_cached
            try:
                await asyncio.to_thread(
                    analysis_service_cached.get_seasonal_heatmap,
                    resolution.mars_year,
                    variable="o3col",
                )
                await asyncio.to_thread(
                    analysis_service_cached.get_seasonal_bands,
                    resolution.mars_year,
                )
            except Exception:
                logger.exception("personal analysis cache warm failed for user %s", user_id)

        final_status = await personal_source_service.get_build_status(user_id)
        await personal_source_service._upsert_build_state(
            user_id=user_id,
            signature_hash=final_status.get("signature_hash") or status.get("signature_hash") or "",
            status="ready",
            stage="ready",
            progress=100.0,
            stage_message=personal_source_service._build_stage_message("ready"),
            error=None,
            duration_ms=final_status.get("duration_ms"),
            built_at=datetime.now(timezone.utc),
        )

    async def personal_cache_rebuild_worker() -> None:
        while True:
            uid = await personal_cache_rebuild_queue.get()
            try:
                await personal_source_service.build_user_cache(uid)
                await warm_personal_runtime_caches(uid)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("personal cache rebuild worker failed for user %s", uid)
            finally:
                personal_cache_pending_users.discard(uid)
                personal_cache_rebuild_queue.task_done()

    app.state.personal_cache_rebuild_queue = personal_cache_rebuild_queue
    app.state.personal_cache_pending_users = personal_cache_pending_users
    app.state.enqueue_personal_cache_rebuild = enqueue_personal_cache_rebuild
    app.state.personal_cache_rebuild_worker_task = asyncio.create_task(personal_cache_rebuild_worker())

    # 数据治理服务（资产总览 / 质量评分 / 血缘信息）
    data_governance_service = DataGovernanceService()
    app.state.data_governance_service = data_governance_service

    # 数据集注册表（只读目录；构造不读取文件，首次查询时才校验小包）
    app.state.dataset_registry = DatasetRegistry(
        EARTH_MERRA2_DIR, earth_dataset_id="earth_merra2_daily_v2",
        legacy_earth_package_dir=EARTH_MERRA2_V1_DIR,
        earth_3hourly_package_dir=EARTH_MERRA2_3HOURLY_DIR,
        earth_3hourly_5deg_package_dir=EARTH_MERRA2_3HOURLY_5DEG_DIR,
        verification_cache_dir=EARTH_TRAINING_CACHE_DIR,
    )
    app.state.default_earth_dataset_id = DEFAULT_EARTH_DATASET_ID
    # 二维地球总览数值服务（复用同一注册表实例与已验证快照）
    app.state.earth_overview_service = EarthOverviewService(app.state.dataset_registry)
    # 地球科研分析服务（季节/纬带/空间异常/极区动力学，同一注册表与快照）
    app.state.earth_research_service = EarthResearchService(app.state.dataset_registry)
    # 同步处理器通过它把 Copilot 协程调度回事件循环（AI 解读端点使用）
    app.state.event_loop = asyncio.get_running_loop()

    # Verify the active Earth release in the background; retired daily packages
    # are never warmed or used as a fallback.
    async def _prewarm_earth_release() -> None:
        registry = app.state.dataset_registry
        for dataset_id in (app.state.default_earth_dataset_id,):
            try:
                descriptor = await asyncio.to_thread(registry.get_dataset, dataset_id)
            except Exception:  # pragma: no cover - 预热失败只记录，不影响服务
                logger.warning("Earth dataset prewarm failed for %s", dataset_id, exc_info=True)
                continue
            logger.info(
                "Earth dataset prewarmed: id=%s availability=%s version=%s",
                descriptor.get("dataset_id"),
                descriptor.get("availability"),
                descriptor.get("dataset_version"),
            )

    app.state.earth_prewarm_task = asyncio.create_task(_prewarm_earth_release())

    # 2. 领域服务：可视化与 ML 数据准备
    logger.info("[2/4] 初始化分析与 ML 准备服务...")
    analysis_service = AnalysisService(data_service)
    app.state.analysis_service = analysis_service

    mcd_overview_analysis_service = AnalysisService(
        mcd_overview_service,
        mcd_variables=OVERVIEW_MCD_VARIABLES,
    )
    app.state.mcd_overview_analysis_service = mcd_overview_analysis_service

    # 3. 核心计算模型：分析变换与训练模型推理
    logger.info("[3/4] 初始化分析变换与训练模型推理服务...")
    analysis_transforms = AnalysisTransforms(data_service)
    app.state.analysis_transforms = analysis_transforms

    app.state.training_inference_service = InferenceService()

    # 4. 初始化 AI 服务
    logger.info("[4/4] 初始化 AI 解读与 Copilot 服务...")
    ai_service = AIService()
    app.state.ai_service = ai_service
    copilot_service = CopilotService()
    app.state.copilot_service = copilot_service

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"  启动完成! 耗时 {elapsed:.1f}s")
    logger.info(f"  数据: {data_service.get_available_years()}")
    logger.info(f"  设备: {app.state.training_inference_service.device}")
    logger.info(f"  API 文档: http://localhost:8000/docs")
    logger.info("=" * 60)

    yield  # ← 应用运行中

    # 关闭时清理
    logger.info("正在关闭服务...")
    await training_router_module.training_service.stop()
    mcd_worker_task = getattr(app.state, "mcd_cache_worker_task", None)
    if mcd_worker_task is not None:
        mcd_worker_task.cancel()
        with suppress(asyncio.CancelledError):
            await mcd_worker_task
    worker_task = getattr(app.state, "personal_cache_rebuild_worker_task", None)
    if worker_task is not None:
        worker_task.cancel()
        with suppress(asyncio.CancelledError):
            await worker_task
    earth_prewarm_task = getattr(app.state, "earth_prewarm_task", None)
    if earth_prewarm_task is not None:
        earth_prewarm_task.cancel()
        with suppress(asyncio.CancelledError):
            await earth_prewarm_task
    await ai_service.close()
    await copilot_service.close()


# ─── 创建 FastAPI 应用 ───

app = FastAPI(
    title="AstraAtmos API",
    description="AstraAtmos — 行星大气预测实验平台后端",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)

# ─── CORS 中间件（允许前端 localhost:5173 访问） ───

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",    # Vite 开发服务器
        "http://localhost:3000",    # 备用端口
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://astraatmos.xyz",   # 允许你的穿透域名
        "https://astraatmos.xyz",  # 如果开启了 HTTPS 也要加上
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 注册路由 ───

app.include_router(analysis.router,                  prefix=API_PREFIX)
app.include_router(predict.router,                   prefix=API_PREFIX)
app.include_router(ai.router,                        prefix=API_PREFIX)
app.include_router(copilot.router,                   prefix=API_PREFIX)
app.include_router(auth.router,                      prefix=API_PREFIX)
app.include_router(upload_router_module.router,        prefix=API_PREFIX)
app.include_router(governance_router_module.router,    prefix=API_PREFIX)
app.include_router(notification_router_module.router,  prefix=API_PREFIX)
app.include_router(user_data_router_module.router,     prefix=API_PREFIX)
app.include_router(feedback_router_module.router,      prefix=API_PREFIX)
app.include_router(training_router_module.router,        prefix=API_PREFIX)
app.include_router(user_models_router_module.router,      prefix=API_PREFIX)
app.include_router(datasets_router_module.router,         prefix=API_PREFIX)
app.include_router(earth_overview_router_module.router,   prefix=API_PREFIX)
app.include_router(earth_analysis_router_module.router,   prefix=API_PREFIX)
app.include_router(earth_predict_router_module.router,    prefix=API_PREFIX)
app.include_router(research_export_router_module.router,  prefix=API_PREFIX)


# ─── 健康检查 ───

@app.get("/")
async def root():
    if FRONTEND_DIST_DIR is not None:
        return FileResponse(FRONTEND_DIST_DIR / "index.html")
    return {
        "name": "AstraAtmos API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if FRONTEND_DIST_DIR is not None:
    _RESERVED_PATHS = {"api", "docs", "redoc", "openapi.json", "health"}

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        normalized = (full_path or "").lstrip("/")
        if normalized in _RESERVED_PATHS or normalized.startswith("api/"):
            return ORJSONResponse({"detail": "Not Found"}, status_code=404)

        if not normalized:
            return FileResponse(FRONTEND_DIST_DIR / "index.html")

        candidate = (FRONTEND_DIST_DIR / normalized).resolve()
        try:
            candidate.relative_to(FRONTEND_DIST_DIR)
        except ValueError:
            return ORJSONResponse({"detail": "Not Found"}, status_code=404)

        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST_DIR / "index.html")
