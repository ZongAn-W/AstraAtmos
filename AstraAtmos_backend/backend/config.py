"""
AstraAtmos 后端配置常量
"""

import os
from pathlib import Path
from typing import Optional, Union
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# ─── 路径 ───
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OPENMARS_DIR = DATA_DIR / "openmars"


def resolve_mcd_dir(
    data_dir: Path,
    configured_path: Optional[Union[str, os.PathLike]],
) -> Path:
    if configured_path and str(configured_path).strip():
        return Path(configured_path)

    try:
        directories = {
            child.name: child
            for child in data_dir.iterdir()
            if child.is_dir()
        }
    except OSError:
        directories = {}

    return directories.get("mcd") or directories.get("MCD") or data_dir / "mcd"


MCD_DIR = resolve_mcd_dir(DATA_DIR, os.getenv("ASTRAATMOS_MCD_DIR"))
MCD_OVERVIEW_DIR = DATA_DIR / "mcd_overview"
MCD_RAW_3H_DIR = Path(os.getenv(
    "MCD_RAW_3H_DIR",
    DATA_DIR / "MCD_Output_global_10m_ls_lst",
))
NOMAD_DIR = DATA_DIR / "nomad"
# Registered Earth compact release directory; a relative value resolves against
# the backend start directory. The path itself is never returned by the API.
EARTH_MERRA2_DIR = Path(os.getenv(
    "ASTRAATMOS_EARTH_MERRA2_DIR",
    DATA_DIR / "earth" / "merra2_daily_v2",
)).expanduser()
EARTH_MERRA2_V1_DIR = Path(os.getenv(
    "ASTRAATMOS_EARTH_MERRA2_V1_DIR",
    DATA_DIR / "earth" / "merra2_daily_v1",
)).expanduser()
# Independent three-hourly release; never reinterpret a daily package path.
EARTH_MERRA2_3HOURLY_DIR = Path(os.getenv(
    "ASTRAATMOS_EARTH_MERRA2_3HOURLY_DIR",
    DATA_DIR / "earth" / "merra2_3hourly_v1",
)).expanduser()
EARTH_MERRA2_3HOURLY_5DEG_DIR = Path(os.getenv(
    "ASTRAATMOS_EARTH_MERRA2_3HOURLY_5DEG_DIR",
    DATA_DIR / "earth" / "merra2_3hourly_5deg_v1",
)).expanduser()

EARTH_DEFAULT_DATASET_IDS = frozenset({
    "earth_merra2_3hourly_v1",
})


def resolve_default_earth_dataset_id(value: Optional[str]) -> str:
    dataset_id = (value or "earth_merra2_3hourly_v1").strip().lower()
    if dataset_id not in EARTH_DEFAULT_DATASET_IDS:
        raise ValueError(
            "ASTRAATMOS_DEFAULT_EARTH_DATASET_ID must name a supported Earth entry dataset"
        )
    return dataset_id


DEFAULT_EARTH_DATASET_ID = resolve_default_earth_dataset_id(
    os.getenv("ASTRAATMOS_DEFAULT_EARTH_DATASET_ID")
)
# Scratch volumes stay outside the Git checkout; interrupted caches are kept.
EARTH_TRAINING_CACHE_DIR = Path(os.getenv(
    "ASTRAATMOS_EARTH_TRAINING_CACHE_DIR",
    BASE_DIR.parents[2] / "earth_training_cache",
)).expanduser()
TRAINING_SCRIPTS_DIR = BASE_DIR / "models" / "training_scripts"
TRAINING_RESULTS_DIR = BASE_DIR / "models" / "training_results"
TRAINING_LOGS_DIR = BASE_DIR / "models" / "training_logs"
PERSONAL_CACHE_DIR = DATA_DIR / "personal_cache"
MCD_CACHE_DIR = DATA_DIR / "mcd_cache"
MOLA_TOPOGRAPHY_PATH = Path(os.getenv(
    "ASTRAATMOS_MOLA_TOPOGRAPHY_PATH",
    DATA_DIR / "assets" / "mola_topography_5deg.nc",
))
# Independent Earth atmospheric lower-boundary asset. Loaded only by models
# explicitly declaring derived_inputs.topography; never a Mars/MOLA fallback.
EARTH_TOPOGRAPHY_PATH = Path(os.getenv(
    "ASTRAATMOS_EARTH_TOPOGRAPHY_PATH",
    DATA_DIR / "assets" / "earth_etopo2022_surface_0p75deg.nc",
)).expanduser()
EARTH_TOPOGRAPHY_5DEG_PATH = Path(os.getenv(
    "ASTRAATMOS_EARTH_TOPOGRAPHY_5DEG_PATH",
    DATA_DIR / "assets" / "earth_etopo2022_surface_5deg.nc",
)).expanduser()

# ─── 网格 ───
SUPPORTED_MARS_YEARS = [27, 28]
DEFAULT_MARS_YEAR = 27
N_LAT = 36
N_LON = 72
OVERVIEW_OZONE_MATCH_TOLERANCE_LS = 2.5
NOMAD_MATCH_TOLERANCE_LS = 2.5

MCD_VARIABLES = [
    "U_Wind", "V_Wind", "Dust_Optical_Depth",
    "Solar_Flux_DN", "Temperature",
]
OVERVIEW_MCD_VARIABLES = [
    var for var in MCD_VARIABLES
    if var != "Dust_Optical_Depth"
]

# 模型训练时的物理主序 (决定了 Tensor 的堆叠顺序)
# 注意: 训练脚本中 Temperature 往往排在 Dust 和 Solar 之前
TRAINING_MASTER_ORDER = [
    "U_Wind", "V_Wind", "Temperature", 
    "Dust_Optical_Depth", "Solar_Flux_DN",
]

VARIABLE_NAMES_CN = {
    "o3col": "臭氧柱浓度", "U_Wind": "纬向风", "V_Wind": "经向风",
    "Temperature": "温度",
    "Dust_Optical_Depth": "沙尘光学厚度", "Solar_Flux_DN": "太阳下行辐射通量",
}

# ─── 缓存 ───
CACHE_MAX_SIZE = 128
CACHE_TTL = 3600

# ─── 降采样 ───
MAX_LS_POINTS = 500  # 热力图/折线图 Ls 维度最大点数

# ─── 纬度带 ───
LATITUDE_BANDS = [
    {"name": "Polar North (60N-90N)", "lat_min": 60, "lat_max": 90},
    {"name": "Mid-Lat North (30N-60N)", "lat_min": 30, "lat_max": 60},
    {"name": "Equatorial (30S-30N)", "lat_min": -30, "lat_max": 30},
    {"name": "Mid-Lat South (30S-60S)", "lat_min": -60, "lat_max": -30},
    {"name": "Polar South (60S-90S)", "lat_min": -90, "lat_max": -60},
]

# ─── API ───
API_PREFIX = "/api"

# ─── AI（Phase 3） ───
AI_API_URL = os.getenv("AI_API_URL", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
AI_MODEL_NAME = os.getenv("AI_MODEL_NAME", "gemini-1.5-flash")
AI_API_KEY = os.getenv("AI_API_KEY", "")

# ─── 上传 ───
USER_UPLOADS_DIR   = DATA_DIR / "user_uploads"
PENDING_REVIEW_DIR = DATA_DIR / "pending_review"
APPROVED_DIR       = DATA_DIR / "approved"
USER_MODELS_DIR = DATA_DIR / "user_models"
TRAINING_WEIGHTS_DIR = DATA_DIR / "training_weights"
MAX_UPLOAD_SIZE_MB = int(os.getenv("ASTRAATMOS_MAX_UPLOAD_SIZE_MB", "512"))
ALLOWED_NC_EXTENSIONS = [".nc", ".nc4", ".netcdf"]
MAX_USER_MODEL_SIZE_KB = int(os.getenv("MAX_USER_MODEL_SIZE_KB", "256"))
ALLOWED_USER_MODEL_EXTENSIONS = [".py"]
USER_MODEL_VALIDATION_TIMEOUT_SECONDS = int(os.getenv("USER_MODEL_VALIDATION_TIMEOUT_SECONDS", "120"))
if USER_MODEL_VALIDATION_TIMEOUT_SECONDS <= 0:
    raise ValueError("USER_MODEL_VALIDATION_TIMEOUT_SECONDS must be a positive integer")
USER_MODEL_FULL_GRID_VALIDATION_TIMEOUT_SECONDS = int(os.getenv(
    "USER_MODEL_FULL_GRID_VALIDATION_TIMEOUT_SECONDS", "900"
))
if USER_MODEL_FULL_GRID_VALIDATION_TIMEOUT_SECONDS <= 0:
    raise ValueError("USER_MODEL_FULL_GRID_VALIDATION_TIMEOUT_SECONDS must be a positive integer")
MAX_TRAINING_WEIGHT_SIZE_MB = int(os.getenv("MAX_TRAINING_WEIGHT_SIZE_MB", "500"))
USER_MODEL_TRAINING_TIMEOUT_SECONDS = int(os.getenv("USER_MODEL_TRAINING_TIMEOUT_SECONDS", "7200"))

# ─── 数据库 ───
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite+aiosqlite:///{DATA_DIR / 'astraatmos.db'}",
)

# ─── 认证 ───
import secrets
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY") or secrets.token_hex(32)
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))
DEFAULT_ADMIN_EMAIL = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@astraatmos.com")
DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin123")

# ─── 邮箱 SMTP ───
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.qq.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER", "")           # QQ 邮箱地址，如 123456@qq.com
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")    # QQ 邮箱授权码（非登录密码）
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "AstraAtmos 行星大气实验室")
VERIFICATION_CODE_EXPIRE_MINUTES = 10             # 验证码有效期（分钟）
VERIFICATION_CODE_COOLDOWN_SECONDS = 60           # 发送冷却时间（秒）

# ─── 训练环境 ───
import sys
TRAINING_PYTHON_PATH = os.getenv("TRAINING_PYTHON_PATH", sys.executable)
