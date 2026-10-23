# 地球日频数据集停用

2026-10-08 起，项目的地球总览、年度分析、训练与历史回测统一使用 `earth_merra2_3hourly_v1`。开发与生产默认一致；三小时发布不可用时直接显示原因，不回退到日频数据包。

## 当前入口

- `GET /api/datasets` 仅列出 `openmars_mcd`、`mcd_overview`、`earth_merra2_3hourly_v1`，默认 Earth ID 为三小时。
- 地球训练页只有三小时选项，默认窗口 56→24、240×480、UTC，支持官方 DLinear 与独立 v1 契约上传模型。
- 地球预测选择器仅提供已完成且权重有效的三小时任务；旧日频结果页保留指标与产物，但禁用预测、比较和复制配置。
- `earth_merra2_daily_v1` / `earth_merra2_daily_v2` 的目录详情、数据快照、总览/分析、训练与地球历史预测请求返回 409 `dataset_retired`。训练创建任务前拒绝；遗留队列在启动子进程前拒绝。
- 年度分析的日均序列仍由三小时源的每日八步聚合产生。这是图表统计口径，不是继续使用日频发布。

## 历史数据

日频文件、数据库记录、日志、checkpoint 及其识别/解析代码保留，用于追溯旧实验。停用不删除文件、不重写任务、不将旧模型当作三小时模型；需要重新创建三小时训练任务。日频底层解析和旧协议文档属于归档资源，不代表对外入口仍可运行。

## 部署

后端工作目录为 `AstraAtmos_backend/backend`。配置 `ASTRAATMOS_EARTH_MERRA2_3HOURLY_DIR` 指向同时包含 `manifest.json` 与 `earth_merra2_3hourly.nc` 的已验证发布目录；相对路径按后端工作目录解析。

`ASTRAATMOS_DEFAULT_EARTH_DATASET_ID` 可省略或设为 `earth_merra2_3hourly_v1`；旧日频值须修改或移除，否则启动会明确报配置错误。无需改写日频路径或删除旧包。前端源码修改后在 `frontend/` 执行 `npm run build`，重启生产前端并完整刷新页面。

## 验证入口

后端停用契约为 `tests/test_earth_dataset_retirement.py`，覆盖目录、默认值、直接请求、缺少 checkpoint 时的预测拒绝、创建任务及启动子进程前拒绝，并核对无身份列的旧任务从超参数识别日频。活动三小时链路继续由 `test_earth_3hourly_*` 覆盖。测试使用项目指定解释器与新的英文临时目录，旧日频成功运行测试属于已停用能力的历史契约。

前端相关测试为 `earthTrainingConfig.test.js`、`earthTrainingDatasetStructure.test.js`、`earthPredictModel.test.js`、`earthOverviewModel.test.js`、`experimentCenterModel.test.js`，覆盖旧目录/草稿/任务无法重新打开日频入口、三小时默认值和选择器。
