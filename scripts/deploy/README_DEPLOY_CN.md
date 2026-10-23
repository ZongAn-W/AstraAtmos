# AstraAtmos Linux 一键部署说明

平台统一使用 AstraAtmos 命名；本次为硬切换，部署目录、脚本、服务名与环境变量前缀均已更新，旧部署配置需按新命名迁移，旧数据与 checkpoint 不兼容。

适用环境：Ubuntu 22.04 GPU 服务器，不依赖 Docker。

## 首次部署

```bash
cd /home/ubuntu/AstraAtmos
bash scripts/deploy/astraatmos.sh install
```

脚本会安装系统依赖、Node.js、Python 虚拟环境、PyTorch、后端依赖，构建前端，并创建 `astraatmos.service`。

## 上传覆盖后重新部署

```bash
cd /home/ubuntu/AstraAtmos
bash scripts/deploy/astraatmos.sh deploy
```

适合你在本地修改项目后，把文件上传覆盖到服务器，再快速重建和重启。

## Git 更新后部署

```bash
cd /home/ubuntu/AstraAtmos
bash scripts/deploy/astraatmos.sh update
```

该命令会执行 `git pull --ff-only`，然后自动走重新部署流程。

## 常用运维命令

```bash
bash scripts/deploy/astraatmos.sh status
bash scripts/deploy/astraatmos.sh logs
bash scripts/deploy/astraatmos.sh health
bash scripts/deploy/astraatmos.sh restart
bash scripts/deploy/astraatmos.sh stop
bash scripts/deploy/astraatmos.sh backup
```

## 访问地址

默认端口是 `8000`：

```text
http://服务器公网IP:8000
```

健康检查：

```text
http://服务器公网IP:8000/health
```

## 环境变量

后端环境文件位于：

```text
AstraAtmos_backend/backend/.env
```

首次部署时脚本会保留已有 `.env`。如果不存在，则从 `.env.example` 创建，并补充部署需要的默认值。

重要配置：

```env
JWT_SECRET_KEY=部署时自动生成或自行设置
AI_API_KEY=你的 AI 接口密钥
MCD_RAW_3H_DIR=/home/ubuntu/Data/MCD_Output_global_10m_ls_lst
TRAINING_PYTHON_PATH=/home/ubuntu/AstraAtmos/AstraAtmos_backend/backend/.venv/bin/python
```

## 数据保留

重新部署不会主动删除：

- `AstraAtmos_backend/backend/.env`
- `AstraAtmos_backend/backend/data`
- `AstraAtmos_backend/backend/models`
- `AstraAtmos_backend/backend/logs`

## 备份

```bash
bash scripts/deploy/astraatmos.sh backup
```

备份默认保存到：

```text
/home/ubuntu/astraatmos-backups/
```

## 排错

查看服务状态：

```bash
systemctl status astraatmos --no-pager
```

查看实时日志：

```bash
journalctl -u astraatmos -f
```

如果前端页面没有更新，重新运行：

```bash
bash scripts/deploy/astraatmos.sh deploy
```

如果 AI 对话不可用，检查 `.env` 中的 `AI_API_KEY`。
