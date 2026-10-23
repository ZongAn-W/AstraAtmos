#!/usr/bin/env bash
set -Eeuo pipefail

SERVICE_NAME="astraatmos"
SERVICE_UNIT_NAME="astraatmos.service"
SERVICE_FILE="/etc/systemd/system/${SERVICE_UNIT_NAME}"
DEFAULT_APP_PORT=8000
APP_PORT="${ASTRAATMOS_PORT:-${DEFAULT_APP_PORT}}"
NODE_MAJOR=20
NODE_MAJOR="${ASTRAATMOS_NODE_MAJOR:-${NODE_MAJOR}}"
PYTORCH_CUDA_INDEX="${PYTORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"
DEFAULT_TORCH_SPEC="torch==2.5.1"
PYTORCH_SPEC="${PYTORCH_SPEC:-${DEFAULT_TORCH_SPEC}}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRONTEND_DIR="${REPO_ROOT}/frontend"
BACKEND_DIR="${REPO_ROOT}/AstraAtmos_backend/backend"
VENV_DIR="${BACKEND_DIR}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
PIP_BIN="${VENV_DIR}/bin/pip"
ENV_FILE="${BACKEND_DIR}/.env"
ENV_EXAMPLE="${BACKEND_DIR}/.env.example"
FRONTEND_DIST="${BACKEND_DIR}/frontend_dist"
BACKUP_ROOT="${ASTRAATMOS_BACKUP_ROOT:-${HOME}/astraatmos-backups}"
MCD_RAW_DEFAULT="${ASTRAATMOS_MCD_RAW_DIR:-${HOME}/Data/MCD_Output_global_10m_ls_lst}"
HEALTH_URL="http://127.0.0.1:${APP_PORT}/health"

log() { printf '[astraatmos] %s\n' "$*"; }
warn() { printf '[astraatmos][WARN] %s\n' "$*" >&2; }
die() { printf '[astraatmos][ERROR] %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<USAGE
Usage: bash scripts/deploy/astraatmos.sh <command>

Commands:
  install   Install system/runtime dependencies, build, install systemd service, start
  deploy    Reinstall app dependencies, rebuild frontend, restart service, health-check
  update    Run git pull --ff-only, then deploy
  restart   Restart the systemd service
  stop      Stop the systemd service
  status    Show systemd service status
  logs      Follow systemd logs
  health    Check http://127.0.0.1:${APP_PORT}/health
  backup    Archive runtime state under ${BACKUP_ROOT}
  help      Show this help
USAGE
}

require_project_root() {
  [[ -d "${FRONTEND_DIR}" ]] || die "Missing frontend dir: ${FRONTEND_DIR}"
  [[ -d "${BACKEND_DIR}" ]] || die "Missing backend dir: ${BACKEND_DIR}"
  [[ -f "${BACKEND_DIR}/requirements.txt" ]] || die "Missing backend requirements.txt"
  [[ -f "${FRONTEND_DIR}/package.json" ]] || die "Missing frontend package.json"
}

have_cmd() { command -v "$1" >/dev/null 2>&1; }

sudo_cmd() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

apt_get() {
  sudo_cmd env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get \
    -o Dpkg::Options::=--force-confdef \
    -o Dpkg::Options::=--force-confold \
    "$@"
}

install_system_packages() {
  log "Installing system packages"
  apt_get update
  apt_get install -y curl ca-certificates git python3 python3-venv python3-pip build-essential pkg-config gnupg
}

verify_node_runtime() {
  have_cmd node || die "Node.js installation did not provide node."
  have_cmd npm || die "npm is not available after Node.js installation."

  local major
  major="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
  [[ "${major}" -ge 18 ]] || die "Node.js installation did not provide Node.js >= 18; got $(node --version 2>/dev/null || echo unknown)."
}

install_node() {
  if have_cmd node && have_cmd npm; then
    local major
    major="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
    if [[ "${major}" -ge 18 ]]; then
      log "Node.js $(node --version) already available"
      return
    fi
  fi

  log "Installing Node.js ${NODE_MAJOR}.x"
  sudo_cmd install -d -m 0755 /etc/apt/keyrings /etc/apt/sources.list.d
  curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | sudo_cmd env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a bash -
  apt_get install -y nodejs
  verify_node_runtime
}

create_venv() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    log "Creating Python virtual environment"
    python3 -m venv "${VENV_DIR}"
  fi
  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
}

install_python_deps() {
  log "Installing PyTorch ${PYTORCH_SPEC} from ${PYTORCH_CUDA_INDEX}"
  "${PIP_BIN}" install "${PYTORCH_SPEC}" --index-url "${PYTORCH_CUDA_INDEX}"

  log "Installing backend dependencies"
  local req_tmp
  req_tmp="$(mktemp)"
  grep -vE '^torch==' "${BACKEND_DIR}/requirements.txt" > "${req_tmp}"
  "${PIP_BIN}" install --index-url "${PYPI_INDEX_URL}" -r "${req_tmp}"
  rm -f "${req_tmp}"
}

ensure_env_key() {
  local key="$1"
  local value="$2"
  if ! grep -qE "^${key}=" "${ENV_FILE}" 2>/dev/null; then
    printf '%s=%s\n' "${key}" "${value}" >> "${ENV_FILE}"
  fi
}

ensure_env() {
  if [[ ! -f "$ENV_FILE" ]]; then
    log "Creating .env from template"
    if [[ -f "${ENV_EXAMPLE}" ]]; then
      cp "${ENV_EXAMPLE}" "${ENV_FILE}"
    else
      touch "${ENV_FILE}"
    fi
  fi

  local secret
  secret="$("${PYTHON_BIN}" - <<'PY'
import secrets
print(secrets.token_hex(32))
PY
)"
  ensure_env_key "JWT_SECRET_KEY" "${secret}"
  ensure_env_key "DEFAULT_ADMIN_EMAIL" "admin@astraatmos.com"
  ensure_env_key "DEFAULT_ADMIN_PASSWORD" "admin123-change-me"
  ensure_env_key "AI_API_KEY" ""
  ensure_env_key "MCD_RAW_3H_DIR" "${MCD_RAW_DEFAULT}"
  ensure_env_key "TRAINING_PYTHON_PATH" "${PYTHON_BIN}"

  if ! grep -qE '^AI_API_KEY=.' "${ENV_FILE}" 2>/dev/null; then
    warn "AI_API_KEY is empty; AI chat features may be unavailable until configured."
  fi
}

build_frontend() {
  log "Installing frontend dependencies"
  cd "${FRONTEND_DIR}"
  if [[ -f package-lock.json ]]; then
    npm ci
  else
    npm install
  fi

  log "Building frontend"
  npm run build

  log "Publishing frontend dist to backend"
  rm -rf "${FRONTEND_DIST}.new"
  mkdir -p "${FRONTEND_DIST}.new"
  cp -a "${FRONTEND_DIR}/dist/." "${FRONTEND_DIST}.new/"
  rm -rf "${FRONTEND_DIST}.old"
  if [[ -d "${FRONTEND_DIST}" ]]; then
    mv "${FRONTEND_DIST}" "${FRONTEND_DIST}.old"
  fi
  mv "${FRONTEND_DIST}.new" "${FRONTEND_DIST}"
  rm -rf "${FRONTEND_DIST}.old"
}

write_service() {
  log "Installing systemd service ${SERVICE_NAME}"
  local service_tmp
  service_tmp="$(mktemp)"
  cat > "${service_tmp}" <<SERVICE
[Unit]
Description=AstraAtmos FastAPI service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${BACKEND_DIR}
Environment=ASTRAATMOS_FRONTEND_DIST=${FRONTEND_DIST}
Environment=ASTRAATMOS_WARMUP_ON_STARTUP=0
ExecStart=${PYTHON_BIN} -m uvicorn main:app --host 0.0.0.0 --port ${APP_PORT:-8000} --workers 1
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE
  sudo_cmd install -m 0644 "${service_tmp}" "${SERVICE_FILE}"
  rm -f "${service_tmp}"
  sudo_cmd systemctl daemon-reload
  sudo_cmd systemctl enable "${SERVICE_NAME}"
}

document_default_uvicorn_command() {
  # Default command: python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
  :
}

restart_service() {
  log "Restarting ${SERVICE_NAME}"
  sudo_cmd systemctl restart "${SERVICE_NAME}"
}

health_check() {
  log "Checking ${HEALTH_URL}"
  local i
  for i in $(seq 1 180); do
    if curl -fsS --max-time 2 "${HEALTH_URL}" >/dev/null; then
      curl -fsS --max-time 5 "${HEALTH_URL}"
      printf '\n'
      return 0
    fi
    sleep 1
  done
  sudo_cmd systemctl status "${SERVICE_NAME}" --no-pager || true
  die "Health check failed after 180 seconds."
}

do_install() {
  require_project_root
  install_system_packages
  install_node
  create_venv
  install_python_deps
  ensure_env
  build_frontend
  write_service
  restart_service
  health_check
}

do_deploy() {
  require_project_root
  create_venv
  install_python_deps
  ensure_env
  build_frontend
  write_service
  restart_service
  health_check
}

do_update() {
  require_project_root
  [[ -d "${REPO_ROOT}/.git" ]] || die "Not a Git checkout: ${REPO_ROOT}"
  cd "${REPO_ROOT}"
  log "Current branch: $(git branch --show-current)"
  git pull --ff-only
  do_deploy
}

do_backup() {
  require_project_root
  mkdir -p "${BACKUP_ROOT}"
  local ts archive
  ts="$(date +%Y%m%d-%H%M%S)"
  archive="${BACKUP_ROOT}/astraatmos-runtime-${ts}.tar.gz"
  log "Creating backup ${archive}"
  tar -czf "${archive}" -C "${BACKEND_DIR}" .env data models logs
  log "Backup written: ${archive}"
}

cmd="${1:-help}"
case "${cmd}" in
  install) do_install ;;
  deploy) do_deploy ;;
  update) do_update ;;
  restart) sudo_cmd systemctl restart "${SERVICE_NAME}" ;;
  stop) sudo_cmd systemctl stop "${SERVICE_NAME}" ;;
  status) sudo_cmd systemctl status "${SERVICE_NAME}" --no-pager ;;
  logs) sudo_cmd journalctl -u "${SERVICE_NAME}" -f ;;
  health) health_check ;;
  backup) do_backup ;;
  help|-h|--help) usage ;;
  *) usage; die "Unknown command: ${cmd}" ;;
esac
