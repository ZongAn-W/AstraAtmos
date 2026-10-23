# 实验中心改版实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` 按任务执行本计划；步骤使用复选框跟踪。该计划只改训练页前端承载方式，不替换训练后端。

**Goal:** 将 `#/training` 从“参数表单 + 日志 + 历史卡片”的长页面改为实验中心：左侧管理实验，右侧按“配置、监控、结果”展示当前实验，并保留现有训练、标签、上传模型、迁移学习、测试和预测能力。

**Architecture:** `ModelTrainingPage.jsx` 继续作为训练控制器，保留 `TrainingContext`、API 请求、表单状态和现有处理函数；新增实验中心外壳及三个右侧工作区组件，把显示职责从控制器中拆出。实验中心阶段由当前任务状态和用户操作通过纯函数推导，不增加草稿实验、备注、收藏或新的后端字段。历史记录继续使用现有任务对象和标签接口，预测入口继续使用 `TRAINING_TASK_HANDOFF_KEY`。

**Tech Stack:** React 19、现有 MUI 图标、现有颜色变量和 CSS、Node.js 内置测试运行器、Vite。首期不增加 UI 框架、状态库、数据库表或后端 API。

---

## 1. 交付后的用户流程

页面地址仍为 `#/training`，全站导航可以显示“实验中心”，训练接口和预测页地址不变。

```text
打开实验中心
  ├─ 左侧：实验目录（全部 / 运行中 / 已完成 / 失败，搜索、标签筛选）
  └─ 右侧：空状态或当前实验工作区

点击“新建实验”
  └─ 配置工作区：数据集 → 模型来源 → 模型结构 → 训练参数 → 命名与标签 → 开始实验

开始训练
  └─ 监控工作区：进度、Epoch、Loss、ETA、实时曲线、日志、停止训练

训练完成
  └─ 结果工作区：结论摘要、指标、Loss、完整配置、模型可用状态、用于预测 / 加入比较 / 复制配置
```

首期的“复制配置”只把已完成任务的配置载入当前页面的新建实验表单，不自动启动训练；“用于预测”沿用现有 sessionStorage handoff；“去模型比较”只导航到现有预测页的模型比较模式，只有在预测页已有可复用的选择 handoff 时才预选任务，不新增比较 API。未实现可持久化的实验备注、独立收藏、实验组和自动生成结论，界面不展示这些入口。

## 2. 文件边界

| 文件 | 变更 | 责任 |
| --- | --- | --- |
| `frontend/src/pages/ModelTrainingPage/experimentCenterModel.js` | 新建 | 阶段推导、任务摘要、状态筛选、指标解析等无副作用纯函数 |
| `frontend/src/pages/ModelTrainingPage/experimentCenterModel.test.js` | 新建 | 纯函数测试，覆盖任务状态、旧任务字段、指标异常和筛选组合 |
| `frontend/src/pages/ModelTrainingPage/ExperimentCenterShell.jsx` | 新建 | 实验中心两栏外壳、页面标题、阶段导航和新建/返回动作 |
| `frontend/src/pages/ModelTrainingPage/ExperimentDirectory.jsx` | 新建 | 左侧实验目录、状态筛选、搜索、标签入口和紧凑实验行 |
| `frontend/src/pages/ModelTrainingPage/ExperimentConfigWorkspace.jsx` | 新建 | 现有训练配置控件的重新编排；不直接请求 API |
| `frontend/src/pages/ModelTrainingPage/ExperimentRunMonitor.jsx` | 新建 | 训练进度、Loss、实时日志、停止按钮和日志滚动状态 |
| `frontend/src/pages/ModelTrainingPage/ExperimentResultPanel.jsx` | 新建 | 完成/失败/停止后的指标、参数、模型状态和后续操作 |
| `frontend/src/pages/ModelTrainingPage/experimentCenter.css` | 新建 | 两栏布局、阶段头部、目录行、工作区响应式规则；不在 JSX 中继续堆叠整页布局样式 |
| `frontend/src/pages/ModelTrainingPage.jsx` | 修改 | 保留业务状态和处理函数，接入外壳与三个工作区，移除重复的旧长页面布局输出 |
| `frontend/src/components/TrainingTags/TrainingHistory.jsx` | 修改 | 提取/复用搜索、标签筛选、批量标签和标签管理逻辑，允许实验目录使用紧凑行渲染 |
| `frontend/src/components/TrainingTags/trainingTags.css` | 修改 | 适配左侧目录宽度、窄屏工具栏和焦点样式 |
| `frontend/src/i18n/zh.js` | 修改 | 增加实验中心阶段、目录、空状态、结果动作和错误文案；保留已有训练字段文案 |
| `frontend/src/i18n/en.js` | 修改 | 与中文键一一对应的英文文案 |
| `README.md` | 修改 | 更新“模型训练”功能描述、模块入口和当前功能边界，说明实验中心是现有训练能力的前端重组 |
| `docs/experiment-center.md` | 新建 | 用户流程、阶段含义、现有能力迁移、未开放边界和验证入口 |

不修改 `AstraAtmos_backend/backend/routers/training.py`、训练 schema、数据库模型或训练子进程。若预测页无法接收“比较预选”而需要后端变更，应先保留“加入模型比较”按钮为导航到预测页并说明当前选择，不能在本计划中顺手扩展预测协议。

## 3. 实施任务

### Task 1: 建立实验中心纯函数契约

**Files:**

- Create: `frontend/src/pages/ModelTrainingPage/experimentCenterModel.js`
- Test: `frontend/src/pages/ModelTrainingPage/experimentCenterModel.test.js`

- [ ] **Step 1: 写失败测试，锁定阶段和摘要契约。** 测试以下实际输入和输出：

```js
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  getExperimentStage,
  filterExperimentTasks,
  parseTaskMetrics,
  buildExperimentSummary,
} from './experimentCenterModel.js';

test('没有活动任务时进入配置阶段', () => {
  assert.equal(getExperimentStage({ activeTask: null, isCreating: false }), 'configure');
});

test('运行中和排队任务进入监控阶段', () => {
  assert.equal(getExperimentStage({ activeTask: { status: 'pending' }, isCreating: false }), 'monitor');
  assert.equal(getExperimentStage({ activeTask: { status: 'running' }, isCreating: false }), 'monitor');
});

test('完成、失败和停止后的任务进入结果阶段', () => {
  for (const status of ['completed', 'failed']) {
    assert.equal(getExperimentStage({ activeTask: { status }, isCreating: false }), 'result');
  }
});

test('新建实验优先于当前已完成任务', () => {
  assert.equal(getExperimentStage({ activeTask: { status: 'completed' }, isCreating: true }), 'configure');
});

test('坏的 metrics JSON 不阻塞结果页', () => {
  assert.deepEqual(parseTaskMetrics('{bad'), {});
  assert.deepEqual(parseTaskMetrics(null), {});
});

test('状态和名称筛选可叠加且不修改原数组', () => {
  const tasks = [
    { id: 1, status: 'completed', custom_model_name: 'Ozone baseline', tags: [{ id: 2 }] },
    { id: 2, status: 'running', custom_model_name: 'Dust run', tags: [] },
  ];
  const result = filterExperimentTasks(tasks, { status: 'completed', search: 'ozone', tagIds: [2] });
  assert.deepEqual(result.map(task => task.id), [1]);
  assert.equal(tasks.length, 2);
});

test('摘要优先使用任务名称、架构、数据集和指标', () => {
  const summary = buildExperimentSummary({
    id: 7,
    custom_model_name: 'Ozone baseline',
    status: 'completed',
    model_source: 'official',
    hyperparameters: JSON.stringify({ model_architecture: 'simvp', training_dataset: 'mcd_overview' }),
    metrics: JSON.stringify({ rmse: 0.0312, mae: 0.0184 }),
  });
  assert.equal(summary.name, 'Ozone baseline');
  assert.equal(summary.architecture, 'simvp');
  assert.equal(summary.dataset, 'mcd_overview');
  assert.equal(summary.metrics.rmse, 0.0312);
});
```

- [ ] **Step 2: 运行测试确认失败。**

Run from `frontend`:

```powershell
node --test src/pages/ModelTrainingPage/experimentCenterModel.test.js
```

Expected: FAIL because the new module and exported functions do not exist.

- [ ] **Step 3: 实现最小纯函数模块。** `getExperimentStage` 只返回 `configure`、`monitor`、`result`；`pending` 和 `running` 映射为 `monitor`，其他未知状态映射为 `result` 以便旧任务可查看。`parseTaskMetrics` 只接受对象或 JSON 对象，数组、空值和非法 JSON 返回 `{}`。`filterExperimentTasks` 组合 `status`、大小写不敏感的 `search`、`tagIds` 交集和 `untagged`。`buildExperimentSummary` 使用已有 `hyperparameters`、`metrics`、`dataset_id`/`training_dataset` 字段，不访问网络、不改变任务对象。

- [ ] **Step 4: 运行测试确认通过。**

```powershell
node --test src/pages/ModelTrainingPage/experimentCenterModel.test.js
```

Expected: all tests pass.

### Task 2: 搭建实验中心外壳和实验目录

**Files:**

- Create: `frontend/src/pages/ModelTrainingPage/ExperimentCenterShell.jsx`
- Create: `frontend/src/pages/ModelTrainingPage/ExperimentDirectory.jsx`
- Create: `frontend/src/pages/ModelTrainingPage/experimentCenter.css`
- Modify: `frontend/src/pages/ModelTrainingPage.jsx`
- Modify: `frontend/src/components/TrainingTags/TrainingHistory.jsx`
- Modify: `frontend/src/components/TrainingTags/trainingTags.css`
- Test: `frontend/src/pages/ModelTrainingPage/experimentCenterShellStructure.test.js`

- [ ] **Step 1: 写结构测试。** 用 `readFileSync` 检查外壳包含 `ExperimentDirectory`、`data-stage`、三个阶段标识 `configure`/`monitor`/`result` 和“新建实验”回调；检查目录包含“全部、运行中、已完成、失败”状态筛选和 `TrainingHistory` 的标签能力。测试同时断言外壳没有调用训练 API，避免展示层重复启动请求。

- [ ] **Step 2: 实现两栏外壳。** `ExperimentCenterShell` 接收以下接口：

```js
{
  stage,
  activeTask,
  tasks,
  isCreating,
  onCreate,
  onSelectTask,
  onBackToDirectory,
  directory,
  config,
  monitor,
  result,
}
```

桌面端使用 `minmax(260px, 300px) minmax(0, 1fr)`；左侧目录固定可滚动，右侧工作区占剩余空间。移动端在 `max-width: 900px` 下改为目录顶部横向工具栏、工作区下方排列。页面头部只放标题、当前阶段、登录状态和“新建实验”，不要把完整训练表单放在标题区。

- [ ] **Step 3: 实现实验目录行。** 每行显示名称、任务 ID、模型架构、数据集、开始时间、状态和一项核心指标；运行中行显示进度。点击行只调用 `onSelectTask(task.id)`。删除、重命名、编辑标签、单条测试等动作通过现有 `TrainingTaskCard` 处理器传入，不能在目录组件内复制 API 调用。

- [ ] **Step 4: 复用标签筛选逻辑。** 将 `TrainingHistory` 的搜索、标签交集、未分组、批量标签和标签管理逻辑保留，增加 `renderMode="directory"` 和 `statusFilter` 参数；目录视觉使用紧凑行，现有完整参数展开仍由结果面板负责。状态筛选应在标签/搜索之后继续生效，改变筛选时清空批量选择。

- [ ] **Step 5: 接入页面但暂时只渲染占位工作区。** `ModelTrainingPage.jsx` 先把已有 `tasks`、`activeTaskId`、`setActiveTaskId`、`loadTasks` 和处理函数接入外壳；右侧三个工作区先显示阶段标题和当前任务名称，确认旧训练提交和轮询没有被重复挂载。

- [ ] **Step 6: 运行结构测试和前端构建。**

```powershell
node --test src/pages/ModelTrainingPage/experimentCenterModel.test.js src/pages/ModelTrainingPage/experimentCenterShellStructure.test.js
npm run build
```

Expected: tests pass and Vite build succeeds.

### Task 3: 迁移配置工作区

**Files:**

- Create: `frontend/src/pages/ModelTrainingPage/ExperimentConfigWorkspace.jsx`
- Modify: `frontend/src/pages/ModelTrainingPage.jsx`
- Modify: `frontend/src/pages/ModelTrainingPage/experimentCenter.css`
- Test: `frontend/src/pages/ModelTrainingPage/experimentConfigStructure.test.js`

- [ ] **Step 1: 写配置结构测试。** 断言配置工作区提供以下顺序：实验名称/标签、数据集、模型来源、模型结构、输入通道、时间窗口、训练超参数、迁移学习、高级设置、配置摘要、开始实验；断言 `ModelSourceSelector`、`UploadedModelPanel`、`DynamicModelParamsForm` 和 `TrainingTaskParameters` 的现有职责没有被新组件重新实现。

- [ ] **Step 2: 定义配置组件接口。** 配置组件只接收数据和回调，接口至少包括：

```js
{
  values: {
    customModelName, newTaskTagIds, trainingDataset, modelSource,
    selectedUploadedModelId, customModelParams, selectedChannels,
    modelArchitecture, useSphere, epochs, batchSize, learningRate,
    windowValue, horizon, earlyStoppingPatience, seed,
    transferEnabled, transferSourceType, transferSourceTaskId,
    transferFreezeMode, finetuneLearningRate,
  },
  resources: { scripts, uploadedModels, trainingWeights, completedTasks },
  validation: { modelNameError, customModelParamErrors, transferStartBlocked },
  actions,
  disabled,
  copy,
}
```

`actions` 直接绑定 `ModelTrainingPage.jsx` 现有 setter 和上传/校验/删除处理函数。组件不得自行调用 `startTrainingTask`；“开始实验”调用父组件的 `onStart`，避免出现两套提交状态。

- [ ] **Step 3: 将表单按三层信息架构重排。** 第一层显示完成一次实验必须填写的字段；第二层将官方/上传模型和架构参数放在“模型结构”；第三层将 `seed`、早停、SPHERE、迁移学习和上传权重放进“高级设置”。默认只展开第一层和模型结构；表单摘要固定显示数据集、模型、输入通道、窗口/步长和轮次/批大小。

- [ ] **Step 4: 保留现有验证和迁移锁定。** 继续使用 `buildTrainingHyperparameters`、`validateCustomModelParams`、`transferSourceConfig` 和 `modelTrainingVisibility`；迁移来源任务选中后必须保持结构锁定，来源失效时恢复快照。上传模型仍要求 `validation_status === 'valid'`，服务器托管数据边界不变。

- [ ] **Step 5: 接入配置工作区并测试。** 页面处于 `configure` 阶段时渲染配置组件；点击目录“新建实验”清空名称、标签和迁移选择，但保留默认模型/数据集参数；配置组件不因目录筛选重新挂载，避免用户编辑时丢失输入。

```powershell
node --test src/pages/ModelTrainingPage/experimentConfigStructure.test.js src/pages/ModelTrainingPage/trainingParamSanitizers.test.js src/pages/ModelTrainingPage/transferSourceConfig.test.js
npm run build
```

### Task 4: 迁移训练监控工作区

**Files:**

- Create: `frontend/src/pages/ModelTrainingPage/ExperimentRunMonitor.jsx`
- Modify: `frontend/src/pages/ModelTrainingPage.jsx`
- Modify: `frontend/src/components/TrainingProgressMonitor.jsx`
- Modify: `frontend/src/components/LossEvolutionChart.jsx`
- Modify: `frontend/src/pages/ModelTrainingPage/experimentCenter.css`
- Test: `frontend/src/pages/ModelTrainingPage/experimentRunMonitorStructure.test.js`

- [ ] **Step 1: 写监控结构测试。** 断言监控组件使用 `TrainingProgressMonitor` 和 `LossEvolutionChart`，显示实时日志、当前任务 ID、停止训练入口、无任务和无日志空状态；断言它不创建新的 interval、WebSocket 或 `fetchLogs` 调用。

- [ ] **Step 2: 实现监控组件。** `ExperimentRunMonitor` 接收 `activeTask`、`progressData`、`logs`、`isProcessing`、`autoScrollPinned`、`onScroll`、`onStop` 和 `copy`。进度区突出百分比、Epoch、当前 Loss、ETA；日志区保留行号、错误/警告/指标色彩和自动滚动提示。将停止按钮只在 `pending`/`running` 状态显示。

- [ ] **Step 3: 让 Loss 图适配工作区。** 为 `LossEvolutionChart` 增加 `compact` 或 `height` 属性，默认行为保持不变；实验中心监控区使用容器宽度和可用高度，不在图表组件里读取窗口宽度。没有 loss 数据时显示“等待训练日志”，不要渲染空坐标轴误导用户。

- [ ] **Step 4: 接入监控阶段。** `TrainingContext` 继续负责 5 秒任务轮询、3 秒日志轮询和 WebSocket；`ModelTrainingPage` 只根据 `activeTask.status` 推导阶段并传递数据。任务状态从 running 变为 completed/failed 后，下一次渲染自动进入结果阶段，保留当前任务选中状态。

### Task 5: 建立结果工作区和后续动作

**Files:**

- Create: `frontend/src/pages/ModelTrainingPage/ExperimentResultPanel.jsx`
- Modify: `frontend/src/pages/ModelTrainingPage.jsx`
- Modify: `frontend/src/pages/ModelTrainingPage/experimentCenter.css`
- Test: `frontend/src/pages/ModelTrainingPage/experimentResultStructure.test.js`

- [ ] **Step 1: 写结果结构测试。** 断言结果组件提供状态摘要、RMSE/MAE/MSE/R²/MAPE/SMAPE（有值才显示）、Loss 图、配置详情、数据集身份、权重可用状态，以及“用于预测、加入模型比较、复制配置、重命名、删除”动作。失败或停止任务必须显示失败原因/metrics note 和重试或复制配置入口，不显示“用于预测”。

- [ ] **Step 2: 实现摘要和指标解析。** 复用 `parseTaskMetrics` 和 `buildExperimentSummary`；指标缺失使用 `--`，非法 JSON 使用“指标暂不可用”状态。模型可用性使用后端返回的 `model_available`，不能仅根据 `status === 'completed'` 判断。

- [ ] **Step 3: 迁移完整参数和既有动作。** 使用 `TrainingTaskParameters` 展开完整超参数；重命名、删除、单任务测试复用现有 `handleRenameTask`、`handleDeleteTask`、`setTestTaskId`。复制配置调用 `readTransferSourceTaskConfig`/已有规范化逻辑把任务配置写入父组件表单状态，同时清空 `activeTaskId` 并进入 `configure`，不触发训练。

- [ ] **Step 4: 保留预测 handoff。** “用于预测”继续调用 `buildTrainingTaskHandoff(task, createUserPredictScope(user?.id))`，写入 `TRAINING_TASK_HANDOFF_KEY` 后导航到 `#/predict?from=training`。动作只对 `completed && model_available` 开放；已失败、已停止或缺失权重的任务显示原因。

- [ ] **Step 5: 处理“加入模型比较”。** 首期调用现有预测导航并携带任务 ID 的前端 handoff；如果当前预测页没有批量预选协议，按钮文案改为“去模型比较”，只导航到已有 `trained_compare` 模式并在页面提示用户选择任务，不伪造选择结果，也不修改后端接口。

### Task 6: 完成页面接线、文案和响应式视觉

**Files:**

- Modify: `frontend/src/pages/ModelTrainingPage.jsx`
- Modify: `frontend/src/i18n/zh.js`
- Modify: `frontend/src/i18n/en.js`
- Modify: `frontend/src/pages/ModelTrainingPage/experimentCenter.css`
- Modify: `frontend/src/components/TrainingTags/trainingTags.css`

- [ ] **Step 1: 收敛页面状态。** 页面只保留一套 `isProcessing`、`activeTaskId`、配置状态、任务轮询和日志轮询。删除旧的左右卡片/历史长列表 JSX，避免旧布局和新布局同时渲染；旧子组件只在新工作区被引用。

- [ ] **Step 2: 更新导航和阶段文案。** 中文至少增加：`实验中心`、`新建实验`、`配置实验`、`训练监控`、`实验结果`、`返回实验目录`、`全部/运行中/已完成/失败`、`复制配置`、`去模型比较`、`模型尚不可用`、`暂无实验`、`等待日志`。英文键保持同一层级；已有 `modelTraining.*` 键不删除，避免其他页面或旧测试失效。

- [ ] **Step 3: 实现视觉层级。** 左侧目录使用低对比背景和状态色点，当前任务使用蓝色边框；右侧工作区使用单一主标题和一个主要动作。配置阶段突出“开始实验”，监控阶段突出“停止训练”，结果阶段突出“用于预测”。不使用三栏卡片墙，不把高级参数和实时日志放在同一屏首层。

- [ ] **Step 4: 完成响应式和无障碍。** 900px 以下目录折叠到顶部；640px 以下所有按钮允许换行，点击区域至少 44px；状态筛选使用可聚焦按钮并暴露 `aria-pressed`；工作区切换后将焦点移到阶段标题；日志保留 `role="log"` 和 `aria-live="polite"`，停止按钮有明确的 `aria-label`。

### Task 7: 同步文档和验证

**Files:**

- Create: `docs/experiment-center.md`
- Modify: `README.md`
- Test: all affected frontend tests

- [ ] **Step 1: 写专题文档。** 记录实验中心三阶段、目录筛选、任务状态来源、现有能力迁移和边界：草稿/备注/收藏/实验组/自动结论未开放；上传个人数据仍不能直接成为训练数据；Earth 训练仍受当前功能边界限制。链接到 `training-model-tags.md`、`earth-dlinear-training.md` 和现有预测文档。

- [ ] **Step 2: 更新 README。** 在“主要功能 → 模型训练”写明实验中心是训练页的新承载方式；在“模块与代码入口”增加四个新组件和 `experimentCenterModel.js`；在“关键业务链路”补充“配置 → 训练上下文 → 监控 → 结果 → 预测 handoff”；在“当前功能边界”说明未新增后端实验实体和未开放能力。

- [ ] **Step 3: 运行前端单元测试和构建。**

```powershell
Set-Location 'frontend'
node --test
npm run build
```

Expected: all existing and new Node tests pass; Vite production build succeeds without unresolved imports.

- [ ] **Step 4: 运行结构检查和 diff 检查。**

```powershell
Set-Location 'AstraAtmos'
git diff --check
rg -n "startTrainingTask|fetchLogs|setInterval|new WebSocket" frontend/src/pages/ModelTrainingPage/Experiment*.jsx
git status --short
```

Expected: API/轮询调用只保留在页面控制器或 `TrainingContext`；新工作区组件不重复启动训练或日志请求；没有意外生成数据文件或锁文件。

- [ ] **Step 5: 浏览器验收四种状态。** 使用已有测试账号或 mock 任务检查：

1. 未选任务：目录和“新建实验”空状态，点击新建进入配置。
2. `pending/running`：自动进入监控，进度、Loss、日志和停止训练可用。
3. `completed && model_available`：结果指标、配置、用于预测和去模型比较可用。
4. `failed` 或完成但无权重：结果页显示原因，禁止预测，复制配置仍可用。

同时检查 1440×900、1024×768、768×900、390×844，深色/浅色主题，中英文，键盘 Tab 和日志自动滚动。若启动本地服务进行浏览器验收，必须按项目约定使用 `$env:CONDA_PREFIX\python.exe` 启动后端，并验证 `/health`、前端首页和 `/api/datasets`；仅做前端静态构建时不需要启动后端。

## 4. 完成定义

- [ ] 默认打开训练页先看到实验目录和空状态/当前实验，不再看到旧的长表单加长历史列表。
- [ ] 配置、监控、结果三种状态可由任务状态稳定推导，刷新页面后运行中任务仍进入监控，历史完成任务仍进入结果。
- [ ] 现有官方模型、上传模型、迁移学习、标签、日志、Loss、重命名、删除、测试和预测 handoff 均可达。
- [ ] 不新增训练后端接口、数据库实体或训练数据能力；Earth 训练和个人上传数据边界保持不变。
- [ ] 旧测试与新测试通过，生产构建成功，四种任务状态和四种屏幕宽度完成浏览器验收。
- [ ] README、专题文档、页面入口和实际源码一致，`git diff --check` 无输出。
