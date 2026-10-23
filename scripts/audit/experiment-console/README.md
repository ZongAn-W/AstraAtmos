# 实验中心验收夹具

本目录的脚本用于**实验中心（`#/training`）的浏览器验收**：验证三列控制台布局、sticky 侧栏、
上传模型主入口、官方架构选择器、底部运行条、响应式无横向溢出，以及训练监控与结果工作区
仍然可用。它们**不属于应用运行时**，只在改版后手动执行。

| 文件 | 作用 |
| --- | --- |
| `seed-probe-profile.mjs` | 建立 / 清理一次性验收档案：临时账号、通过校验的探针上传模型、可选的 running / completed 任务与真实日志文件 |
| `verify-console.mjs` | 61 项控制台检查（布局、sticky、上传模型主入口、官方架构选择器、检查器联动、响应式、可访问性） |
| `verify-stages.mjs` | 21 项阶段检查（监控阶段的进度 / Loss / 终端日志，结果阶段的指标 / 折叠日志 / 复制配置回填） |
| `verify-font-hierarchy.mjs` | 桌面端字体层级、框线、首屏密度与检查器验收：1440×900 / 1920×1080 的字号与颜色读数、静态描边与面层统计、每个分区在运行条上沿以上的可见量、检查器结论 / 问题清单、**信息减法后 16 个已删除节点的确认**、文字裁切与横向溢出检测，以及首屏 / 上传管理展开 / 参数区 / 检查器 / 运行条 / 整页截图 |
| `verify-interaction-states.mjs` | 交互状态视觉验收：用真实鼠标事件触发 hover / focus / 选中 / invalid，记录每个控件的实际描边、背景与 `box-shadow`，确认强描边只出现在真实状态上 |
| `verify-inspector-states.mjs` | 检查器状态视觉验收：分别构造「全部就绪」与「未登录 / 待处理」两种状态，逐项断言状态胶囊、就绪记号、分隔线数与层级色，并输出 `-A-blocked` / `-B-ready` 截图 |
| `verify-directory-empty.mjs` | 左侧实验目录的状态与密度验收：空目录 / 加载失败（注入 500）/ 有内容 / 筛选无匹配四种状态，断言目录高度、筛选栏可见性、空态文案与动作、裁切与溢出 |

## 前置条件

1. **构建前端**：在 `frontend/` 执行 `npm run build`。
2. **后端在跑**：`GET http://127.0.0.1:8000/health` 返回 200（验收会调用任务列表、日志与登录接口）。
3. **生产静态服务在跑**：在仓库根目录执行 `node scripts/serve-prod.mjs`（`ROOT=frontend/dist`、`PORT=5173`、`API_PORT=8000`），
   或任何等价地把 `frontend/dist` 与 `/api` 代理到后端的静态服务。
4. **无头浏览器**：启动带远程调试端口的 Edge/Chrome，例如
   `msedge --headless=new --remote-debugging-port=9333 --user-data-dir=%TEMP%\astraatmos-audit`。

## 执行

```powershell
# 1) 建立验收档案（账号 + 探针上传模型 + running/completed 任务）
node scripts\audit\experiment-console\seed-probe-profile.mjs prepare

# 2) 两套验收（默认读取上一步写出的 .probe-profile.json 里的 JWT）
node scripts\audit\experiment-console\verify-console.mjs --port 9333
node scripts\audit\experiment-console\verify-stages.mjs --port 9333

# 3) 清理：删除账号、上传模型、任务、日志文件与临时文件
node scripts\audit\experiment-console\seed-probe-profile.mjs cleanup
```

只做布局检查时可以用 `--no-tasks` 跳过任务注入；也可以直接用 `--token <jwt>` 指向别的账号。

## 字体层级验收

`verify-font-hierarchy.mjs` 单独用于桌面端（和回归用移动端）的字号核对，不修改任何数据：

```powershell
# 默认两个桌面视口，输出 shots/<label>/ 下的截图与 font-metrics.json
node scripts\audit\experiment-console\verify-font-hierarchy.mjs --label after

# 追加回归视口（1024×768 走桌面断点，390×844 走移动断点）
node scripts\audit\experiment-console\verify-font-hierarchy.mjs --label mobile --viewports 1024x768,390x844

# 用指定样式表覆盖构建产物里的 CSS，用于在同一份构建上对比改前 / 改后的字号与框线
node scripts\audit\experiment-console\verify-font-hierarchy.mjs --label before --css <path-to-old-css>

# 浅色主题核查
node scripts\audit\experiment-console\verify-font-hierarchy.mjs --label light --theme light

# 配置就绪状态（自动把实验名填成唯一值，核对绿色状态句与实心橙主按钮）
node scripts\audit\experiment-console\verify-font-hierarchy.mjs --label ready --ready
```

脚本在配置阶段采集：页头、画布头部、四个分区、专家页签、检查器、运行条与实验目录的
computed `font-size` / 颜色，以及每个容器的四边描边、面层背景与 `box-shadow`；同时扫描
`.experiment-center` 下所有隐藏溢出的叶子节点判断是否出现裁切。
`font-metrics.json` 是逐节点的原始读数，可直接与另一次运行对比。

首屏（`<x>-01-first-screen`）在**收起「管理上传模型」**、`scrollY=0` 且布局稳定后采集：
该面板展开时会多出模型列表与格式说明（约 220px），高度读数会失真；展开态单独记为
`<x>-01b-uploaded-manage`。`groupFirstScreen` 给出每个分区在运行条上沿以上有多少可见
（`full` / 像素数 / `none`），用于核对首屏信息密度。

`--ready` 会在进入配置阶段后把实验名填成 `final_visual_audit` 并等检查器胶囊变为 `ready`，
用于核对「配置就绪」下的运行条主按钮与检查器状态；不改变任何业务状态，只填一个表单字段。
`--official` 先切到官方模型再采集。

已删除的装饰 / 重复节点在 `PROBES` 里保留 key 并标记为 `__removed__`：脚本会报告
「N 个已删除节点确认不在页面上」，把信息减法的结果固化成可复跑的检查，而不是只靠截图。

## 交互状态验收

```powershell
node scripts\audit\experiment-console\verify-interaction-states.mjs --label round2
```

脚本按真实用户路径操作（`Input.dispatchMouseEvent` 按下与点击，不用 `element.focus()`）：
参数输入框静态 / 悬停 / 聚焦，数据集下拉聚焦，模型来源胶囊选中态，官方架构选择器选中态与
选择写回，实验名聚焦与非法（空 / 重名，由组件自身的 `validateModelName` 触发），次要按钮与
目录筛选悬停，运行条贴底。输出 `interaction-states.json` 与 `state-*.png`，用于证明
「强描边只留给选中、聚焦与错误」。

## 检查器状态验收

```powershell
node scripts\audit\experiment-console\verify-inspector-states.mjs --label step4
```

脚本在每个视口跑两遍：**未登录**（存在错误 / 待处理）与**已登录 + 有效上传模型 + 唯一实验名**
（全部就绪）。就绪那一轮会等待登录态与模型列表加载完成，必要时整页刷新重试一次（只重试加载，
不改任何应用状态）。输出 `inspector-states.json` 与 `<视口>-A-blocked.png` / `<视口>-B-ready.png`，
断言状态胶囊、当前模型身份、**正常态一行结论 / 问题态逐条原因与定位入口**、静态分隔线数量、
标签 / 值 / 说明三级颜色、面板宽度与内部滚动，并确认常态摘要与折叠详情已移除。

## 实验目录状态验收

```powershell
# 账号里有任务：核对「有内容」与「筛选无匹配」
node scripts\audit\experiment-console\seed-probe-profile.mjs prepare
node scripts\audit\experiment-console\verify-directory-empty.mjs --label has-content

# 账号里没有任务：核对「空目录」
node scripts\audit\experiment-console\seed-probe-profile.mjs cleanup
node scripts\audit\experiment-console\seed-probe-profile.mjs prepare --no-tasks
node scripts\audit\experiment-console\verify-directory-empty.mjs --label empty
```

脚本按验收账号里有没有任务**自适应断言组**（空目录组或有内容组），两种账号状态都能跑。
每个场景使用**独立页面会话**，因此注入的失败包装不会残留到下一场景；`error` 态通过
`Page.addScriptToEvaluateOnNewDocument` 只让 `GET /api/training/tasks` 返回 500 构造，
不改仓库代码也不改后端。输出 `directory-states.json` 与 `<视口>-A-directory` /
`<视口>-A2-no-match` / `<视口>-B-error` 截图。

## 注意

- **只对本地开发库运行。** `seed-probe-profile.mjs` 会直接写 `data/astraatmos.db` 并创建账号，
  `cleanup` 会删除该账号及其模型、任务、标签与日志。
- 探针模型源码（`.probe-surface-net.py`）与登录 JWT（`.probe-profile.json`）都已加入 `.gitignore`，
  不要提交或外传；`shots/` 下的截图与字号读数同样不入库。
- 目标准确率依赖视口：`verify-console.mjs` 使用 CDP 设置 1440×900 / 1024×768 / 390×844 三档，
  并在改动布局后重新确认 sticky 吸附位置（86px）与运行条贴底位置；`verify-font-hierarchy.mjs`
  默认用 1440×900 与 1920×1080，其余视口通过 `--viewports` 指定。
- 账号里存在运行中的实验时页面会自动进入监控阶段（既有行为），因此 `verify-console.mjs`
  先点“新建实验”回到配置阶段再检查配置相关项。
