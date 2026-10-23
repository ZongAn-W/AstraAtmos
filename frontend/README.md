

# AstraAtmos 行星大气实验室 — 前端本地运行指南

---

## 环境准备（只需一次）

### 1. 安装 Node.js

前往 https://nodejs.org 下载 **LTS 版本**（≥18），安装后打开终端验证：

```bash
node --version    # 应显示 v18.x.x 或更高
npm --version     # 应显示 9.x.x 或更高
```

### 2. 在 PyCharm 中打开项目

1. 用 PyCharm 打开整个 `AstraAtmos/` 目录
2. PyCharm 会自动识别 `frontend/` 为前端子项目
3. 建议安装 PyCharm 的 **JavaScript and TypeScript** 插件（通常自带）

---

## 启动前端（每次开发时）

打开 PyCharm 的 **Terminal**（底部面板），执行：

```bash
# 1. 进入前端目录
cd frontend

# 2. 首次运行需安装依赖（之后不用重复）
npm install

# 3. 启动开发服务器
npm run dev
```

你会看到类似输出：

```
  VITE v6.x.x  ready in 300ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: http://192.168.x.x:5173/
```

**浏览器打开 http://localhost:5173 即可看到页面。**

修改任何 `.jsx` 文件后保存，浏览器会自动热更新，无需手动刷新。

---

## 项目结构说明

```
AstraAtmos/
├── AstraAtmos_backend/
│   └── backend/                 ← FastAPI 后端
│       ├── main.py              ← 应用入口 + lifespan 初始化
│       ├── config.py            ← 全局配置常量
│       ├── auth/                ← JWT 认证模块
│       │   ├── security.py      ← 密码哈希 / Token 生成验证
│       │   └── dependencies.py  ← FastAPI Depends 函数
│       ├── database/
│       │   ├── engine.py        ← SQLAlchemy async engine
│       │   └── models.py        ← User / UploadRecord / Notification / Feedback
│       ├── routers/             ← API 路由
│       │   ├── auth.py          ← /api/auth/*（注册/登录/验证码/找回密码）
│       │   ├── upload.py        ← /api/upload/*（文件上传/贡献/审核）
│       │   ├── notification.py  ← /api/notification/*（站内通知）
│       │   ├── user_data.py     ← /api/user-data/*（用户数据可视化）
│       │   ├── feedback.py      ← /api/feedback/*（用户反馈）
│       │   ├── analysis.py      ← /api/explore/*（内置数据探索）
│       │   ├── predict.py       ← /api/predict/*（预测）
│       │   └── ai.py            ← /api/ai/*（AI 对话）
│       ├── services/            ← 业务逻辑层
│       │   ├── upload_service.py    NC 文件校验 + SHA256 去重
│       │   ├── user_data_service.py 用户数据按需读取 + 热更新
│       │   ├── email_service.py     QQ SMTP 发送验证码
│       │   ├── analysis_service.py  内置数据分析
│       │   └── ai_service.py        LLM 对话
│       └── core/                ← ML 核心模块
│
└── frontend/                    ← React 前端
    ├── index.html               ← HTML 入口（系统字体 fallback，无远程字体请求）
    ├── package.json             ← 依赖声明
    ├── vite.config.js           ← Vite 配置（/api/* → localhost:8000 代理）
    └── src/
        ├── main.jsx             ← 应用挂载点：SettingsProvider > ToastProvider > AuthProvider > App + AuthModal + Toast
        ├── App.jsx              ← 根组件：页面路由（hash） + 全局弹窗状态
        ├── index.css            ← 全局样式 + observation-window CSS 类
        │
        ├── constants/
        │   └── colors.js        ← 颜色 Token（C.mars / C.blue / C.ice 等）
        │
        ├── i18n/
        │   ├── index.js         ← useT() Hook
        │   ├── zh.js            ← 中文文案
        │   └── en.js            ← 英文文案
        │
        ├── utils/
        │   └── colormaps.js     ← 7 套色彩映射（Inferno/Viridis/Plasma/Magma/Cividis/Jet/RdBu）
        │
        ├── hooks/
        │   └── useScrollLock.js ← 模块级 lockCount，支持多层弹窗
        │
        ├── contexts/
        │   ├── SettingsContext.jsx   ← 语言/主题/Colormap/单位制/精度/导出，持久化 localStorage
        │   ├── AuthContext.jsx       ← 登录状态/token/user，自动注入 Bearer 请求头
        │   ├── ToastContext.jsx      ← 全局 Toast 队列，showToast(msg, type)
        │   └── DataOverviewContext.jsx ← DataOverviewPage 内部状态
        │
        ├── components/
        │   ├── Navbar.jsx            ← 导航栏（用户头像下拉/通知铃铛/管理员入口）
        │   ├── SettingsFab.jsx       ← 左下角浮动按钮（语言/主题/Colormap 快捷切换）
        │   ├── SettingsPanel.jsx     ← 右侧完整设置抽屉
        │   ├── AuthModal.jsx         ← 登录/注册/找回密码（根节点挂载，共享弹窗焦点）
        │   ├── NotificationPanel.jsx ← 站内通知抽屉
        │   ├── AdminReviewPanel.jsx  ← 管理员 NC 文件审核抽屉
        │   ├── FeedbackManagePanel.jsx ← 管理员反馈管理抽屉
        │   ├── FeedbackModal.jsx     ← 用户反馈提交弹窗（支持截图）
        │   ├── ContributeModal.jsx   ← 三步贡献向导（createPortal）
        │   ├── ContributeHistoryPanel.jsx ← 贡献记录抽屉
        │   ├── ChangePasswordModal.jsx ← 修改密码（createPortal）
        │   ├── ConfirmDialog.jsx     ← 通用确认对话框（createPortal）
        │   ├── ui/                   ← Button / Panel / Input / Badge / SegmentedControl、弹窗焦点与共享样式
        │   ├── Toast.jsx             ← Toast 通知条（createPortal）
        │   ├── StarField.jsx         ← 星空粒子背景
        │   ├── GlowCard.jsx          ← 发光卡片
        │   ├── SectionTitle.jsx      ← 区块标题
        │   ├── Mars3DPlaceholder.jsx ← Three.js 火星球体（3 光源系统）
        │   └── SphericalFieldCanvas.jsx ← DataOverviewPage 球面场渲染
        │
        ├── pages/
        │   ├── HomePage.jsx          ← 首页（3D 火星 + feature cards）
        │   ├── DataOverviewPage.jsx  ← 数据中心（react-globe.gl + 6 数据窗口）
        │   ├── ExplorePage.jsx       ← 数据探索（含 MyDataTab 用户数据管理）
        │   ├── PredictPage.jsx       ← 已训练模型预测与多模型对比（Canvas 场可视化）
        │   ├── AIPage.jsx            ← AI 对话助手
        │   └── AboutPage.jsx         ← 关于页面
        │
        └── services/
            └── api.js               ← 全部后端调用封装（fetch + Auth header 注入）
```

---

## 界面样式与可访问性

`src/index.css` 统一维护 `--surface-*`、`--text-*`、品牌/状态色、三级边框、圆角、控件高度和字号令牌。深浅主题共用这些语义，旧 `--bg-*` / `--border-*` 为兼容别名。字号通过 `--font-scale` 保留设置缩放；正文优先使用 Noto Sans SC / Inter，展示字体使用 Sora，当前未打包字体文件，实际由系统无衬线 fallback 渲染，不请求 Google Fonts。

首页保留地球、星空与轨道；深色模式的训练和预测页复用首页 `--bg-gradient` 与既有 `StarField`，训练页不再叠加局部深色光晕或实色 body 覆盖，面板、输入及图表继续使用实色表面。浅色主题不显示星点，系统开启“减少动态效果”时星点停止闪烁，数据总览保持实色工作台；鼠标视差与地球交互仅作用于首页。冰蓝表示选择/定位，橙色表示主操作，绿色/红色表示成功/错误，紫色用于诊断。科学色带和 `C.blue` / `C.mars` 兼容常量不受界面状态色调整影响。共享基础控件在 `src/components/ui/Controls.jsx`，新增图标复用既有 MUI Icons，没有新增依赖。

桌面精修沿用既有令牌：导航常态 13px / 500，当前项 600、冰蓝文字和 2px 底部线；中文正文保持 14px，辅助信息至少 12px / 18px 行高。总览关键名称提供完整名称 `title`，主要面板与内部区域分别使用 default / subtle 边框或分隔线，辅助工具保留入口并降低强调。

单模型预测通过 `prediction-display--empty` 显示约 140px 的引导面板，暂不显示三张空结果卡片；有模型时定位已有选择器，无模型时进入现有训练页。禁用的预测动作通过相邻说明和 `aria-describedby` 解释原因。地球与火星的单模型页面共用同一套区块与文案：地球不再输出标题、说明段、`TO3 · DU` 胶囊、上下文折叠块、起点说明行、参考场说明段和「刷新起点范围」按钮，主按钮为共用的 `🚀 开始预测`；完整测试集卡片下的分区行由 `earthSplitRanges`/`buildEarthSplitLabelSource` 与共用的 `getSplitLabel` 渲染。训练页底部条默认约 56px，检查器登录提示为辅助文字，实验登录动作集中在底栏；正文仍根据实测条高预留空间，以适应字号缩放。真实预测的数据、图表、全屏和导出路径继续沿用既有实现。

登录、确认弹窗和设置/通知抽屉使用 `useDialogFocus`：打开后焦点进入，Tab 循环、Esc 关闭，关闭后返回可用触发项；堆叠弹窗只处理最上层。设置快捷菜单支持键盘选择，通知请求失败单独显示错误和重试，不作为空列表显示。

从 `frontend` 目录运行 `npm run build` 构建，前端测试通过 `node --test` 执行 `src` 下的 `*.test.js`。应用运行、后端解释器和生产代理启动方式以[主 README](../README.md)为准。

## 运行后端

```bash
cd AstraAtmos_backend/backend
pip install -r requirements.txt
pip install torch==2.5.1       # CPU；或安装对应 CUDA 版本
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Windows 当前工作区不要在启用 reload 时直接调用 `uvicorn`：Uvicorn 0.34.0 会先建立 Selector 事件循环，无法被项目在导入期设置的 Proactor 策略替换，可能出现 `WinError 10055` 与 `Executor shutdown has been called`。后端按工作区规定的入口 `run_backend.py` 启动，命令、原因与启动后校验见[主 README](../README.md)。

`vite.config.js` 已配置代理：前端所有 `/api/*` 请求自动转发到 `http://localhost:8000`。

**开发时需同时运行两个终端：**
- 终端 1：`cd AstraAtmos_backend/backend && uvicorn main:app --reload`
- 终端 2：`cd frontend && npm run dev`

**环境变量**（`AstraAtmos_backend/backend/.env`，不提交 git）：

```env
AI_API_KEY=your_gemini_api_key
JWT_SECRET_KEY=your_random_secret
SMTP_HOST=smtp.qq.com
SMTP_PORT=465
SMTP_USER=your_qq@qq.com
SMTP_PASSWORD=your_qq_auth_code
```

---

## 常见问题

**Q: `npm install` 报错？**
A: 确认 Node.js 版本 ≥ 18。用 `node --version` 检查。

**Q: 页面空白？**
A: 打开浏览器 F12 控制台看报错信息。

**Q: 如何加新页面？**
A: 在 `pages/` 下创建新文件，在 `App.jsx` 的 `VALID_PAGES` 数组和渲染条件中注册，在 `Navbar.jsx` 中添加导航项。

**Q: 注册/登录报错"无法连接到服务器"？**
A: 确认后端已启动（`uvicorn` 运行在 8000 端口），且 `.env` 中 `JWT_SECRET_KEY` 已设置。

**Q: 验证码发不出去？**
A: 检查 `.env` 中 SMTP 配置；QQ 邮箱需在设置中开启 SMTP 并获取授权码（非登录密码）。

**Q: 弹窗/模态框出现在屏幕角落位置不对？**
A: 检查该组件是否使用了 `ReactDOM.createPortal(..., document.body)`。Navbar 的 `backdrop-filter` 会创建新 containing block，导致子组件 `position:fixed` 失效。
