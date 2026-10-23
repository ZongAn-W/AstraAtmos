# Earth Three-Hour Full-Grid Models v2

An opt-in uploaded-model contract for `earth_merra2_3hourly_v1`. The model receives
the complete global map in one call. Existing v1 tile models remain supported.
Download [the PyTorch template](earth-3hourly-fullgrid-model-template.py) and upload
it as a new model; changing an existing uploaded source invalidates its hash.

Optional `MODEL_SPEC.derived_inputs` works identically for this schema: `ls`
provides per-window `[batch, window]` solar longitude and `topography` provides
finite float32 `[batch,1,240,480]` ETOPO orography in metres on the input device.
The fixed calls are `model(x)`, `model(x, ls)`, `model(x, topography)` and
`model(x, ls, topography)`. Terrain does not add a meteorological channel or a
time axis and is not normalized using meteorological statistics. The server
protects the shared asset from in-place model operations. `season_algorithm`
and terrain content/grid/processing identities are frozen independently when
declared; old checkpoints gain no new fields. Semantics, exact declarations and refusal rules are in
[the v1 contract document](earth-3hourly-uploaded-model.md#季节坐标-derived_inputs).
The [terrain and season template](earth-3hourly-fullgrid-topography-model-template.py)
actually consumes both tensors. The existing SimVP remains independent of terrain.
See [asset construction and validation](earth-topography-asset.md).

| Property | Full-Grid Contract |
| --- | --- |
| Schema | `astraatmos_earth_3hourly_fullgrid_model_v2` |
| Implementation | `astraatmos_earth_3hourly_fullgrid_runner_v2` |
| Input | finite float32 `[B,window,C,240,480]` |
| Output | finite float32 `[B,horizon,1,240,480]`, same input device |
| Channels | TO3 first, then selected U10M, V10M, T2M, SWGDN in order |
| Batch | 1-64 complete global time windows, default 1 |
| Time | 1-240 three-hour UTC steps for window and horizon |
| Normalization | Training-only statistics and frozen task partitions |
| Evaluation | All validation/test windows and global grid cells, DU metrics |

The feed uses the same fields as v1, with the v2 schema and
`spatial_tile_shape=[240,480]`. This retained field name describes the actual call
size; v2 does not spatially subdivide inputs, outputs or diagnostic permutations.
No coordinate arrays or halo arguments are supplied. Optional derived tensors
are explicit arguments outside the meteorological channel list.

The SimVP template has lighter defaults: spatial hidden dimension 8, temporal
hidden dimension 16 and 2 temporal blocks. Original 32/64/3 parameters remain
available. The model receives full maps, but this alone does not guarantee global
receptive fields, physically conserved transport or a spherical solution.
Conv2d longitude padding is circular; latitude padding and the transposed
decoder retain ordinary numerical boundaries. Surface winds are optional inputs.
Start with batch 1 and measure memory for the requested windows before increasing
to 2, 4 or 8. The batch ceiling is the platform parameter limit, not a guarantee
that a model or GPU can hold 64 global windows. The server never
silently reduces resolution, batch or precision when a task runs out of memory.

Upload checks all 16 channel combinations on complete maps, using declared short
windows 2->1 when available, otherwise the first declared values. A terrain
declaration requires the configured verified asset for every probe; a missing or
invalid asset fails only models that declare it, without a zero-field fallback.
Reports record
`validation_scope=declared_channels_probe_windows`; admission is not proof for
every time configuration. Task creation reuses the report saved by upload or
manual revalidation. It checks ownership, valid status, evidence for the concrete
dataset/schema, the source hash, parameter schema and valid batch range, then
freezes the source, schema and parameters and queues the task. It does not repeat
an isolated dry-run for the task's actual window/horizon, channel subset, batch or
parameters. Training still checks outputs and finite gradients, and checkpoint
strict reload still checks eval sample independence with the saved batch.
The upload's isolated CPU check does not guarantee CUDA memory capacity or that
every task configuration can run.
Upload/revalidation begins with `USER_MODEL_VALIDATION_TIMEOUT_SECONDS`
(default 120 seconds). After the isolated child imports the source, normalizes
its dataset declaration and validates its parameter schema, it announces a
full-grid contract to the parent. The parent then uses the larger of the ordinary
budget and `USER_MODEL_FULL_GRID_VALIDATION_TIMEOUT_SECONDS` (default 900 seconds).
The deadline is measured from process start; phase messages cannot reset it.
The parent never imports uploaded source to select a budget. A hung import or
invalid declaration cannot obtain the extension. Both settings must be positive
integers. Upload/revalidation waits for the isolated process in a background
thread so unrelated API requests remain responsive.

Explicit `earth_probe` calls still check the requested complete configuration;
full-grid probes use the same larger budget, with their frozen schema checked
against the executed model declaration. Explicit validator budgets, including
short test timeouts and the in-process `None` override, retain their original
semantics. Tile/Mars checks keep the ordinary budget. Task creation reuses the
saved report and does not run these probes. Unknown or failed saved upload
reports prohibit training. The full 240x480 grid, all 16 channel combinations,
default model parameters and all forward/backward and eval independence checks
remain enabled. A 900-second limit is a bounded CPU validation allowance, not a
promise that every architecture or requested training configuration will fit.

On 2026-10-16, the uploaded shared terrain-affine gate architecture with its
original defaults (16/64/128 hidden dimensions, two temporal blocks, gate width
32) passed the complete isolated CPU float32 upload check in **700.78 seconds**
(two PyTorch threads). This exercised all 16 channel combinations, 2->1 temporal
windows, full 240x480 inputs/outputs, B=1/2, seasonal coordinates, verified ETOPO
terrain, finite gradients and eval sample independence. Its source bytes,
identity, version and display name were preserved; the existing service saved
the successful report after a SQLite backup. This is an upload compatibility
check, not a full training or forecast-accuracy result. Grouped backend regression
runs cover 209 unique tests, including deadline extension/termination, large
reports, composed declarations, responsive upload/revalidation, ownership,
downloads and cached training admission. Run the relevant tests from
`AstraAtmos_backend/backend/` with the configured AstraAtmos interpreter and
`-m pytest --asyncio-mode=auto`; the async service tests need this option.

An available upload report covers its declared short windows, not an arbitrary
20->20 configuration or requested batch. An explicit probe timeout is reported as
`uploaded_model_validation_timeout`: the actual configuration has not finished
validation; it is not proof of incompatibility. The parent
drains the validation result before joining the child so reports larger than the
Windows process pipe do not cause false timeouts. A result alone is insufficient:
the child must exit successfully within the same deadline. Infinite model code
and hung process exit still terminate at the deadline.

Historical 2026-10-10 isolated CPU admission measurements, made while task
creation still ran a configuration probe, used the stored full-grid
SimVP source with five channels, 20->20 and parameters 8/16/2/0.1. Batch 8
completed in 45.46 seconds under the original 120-second budget; batch 32 timed
out after 120.56 seconds, then passed the complete check in 234.64 seconds under
the new default 300-second task budget. The screenshot does not reveal the
requested batch/parameters, so these runs reproduce the budget failure for the
stored model, not its exact screenshot configuration. Larger batches may still
time out or exhaust CPU/GPU memory. This is a budget correction, not a measured
compute speedup; the tests use synthetic tensors and create no training task,
read no release data, upload no model and publish no checkpoint. A separate
large-report regression covers the pipe issue; the stored model's normal report
is smaller than the Windows pipe and does not establish that issue as its cause.
Validation covered 158 upload/admission/tile-regression tests plus seven focused
deadline/explicit-budget/report tests (partly overlapping). No real training was
started; defaults still cannot guarantee that every full-grid configuration fits
an explicit probe's validation budget or device memory. These measurements describe
the previous admission flow; task creation now reuses the saved upload report.

The server freezes the schema in the model reference and
`_earth_uploaded_contract_schema`. Source, task, saved build configuration and
checkpoint schema must agree. The three-hour artifact container stays compatible,
while the v2 model implementation and full spatial call size are independently
recorded. Strict reload and eval independence use full maps. The parent completion
and restart recovery checks use the larger of the generic and global validation
budgets (global default 900 seconds), including the independent 36x72 full-grid
contract. Explicit reload deadlines take priority. V1 spatial tiles retain their
separate 30-second checkpoint gate. The saved-batch CPU round trip and strict
model/weight checks remain unchanged. Already failed timeout-only tasks require
[explicit verified recovery](earth-metrics-task-recovery.md#重载超时恢复).
Old weights are not automatically promoted.

The normalized cache is time-major `[time,C,240,480]`, tagged
`earth_full_grid_v1` with explicit schema/version, data identity and normalization.
Temporal windows are read as needed, without expanding overlapping windows into
a second dataset. Preparation reuses shared validation proofs, train-only statistics, and content-addressed full-grid or tile caches. Changing model parameters, batch size, learning rate, or epochs does not rebuild the normalized volume; changing windows only rebuilds it when the actual training interval changes. Logs show hit/miss status, chunk progress, integrity checks, and stage times. See [preparation cache](earth-preparation-cache.md).
One temporal origin is one sample, rather than 100 spatial samples. This changes
the optimization batch composition even when the loss still weights all pixels.

Historical prediction uses one model call per origin. Diagnostics/PFI process one
complete origin at a time, preserve whole-window channel permutation and DU
metrics, and keep at most the recipient/source windows resident. The final global
evaluation covers all selected leads/cells. V1 continues using spatial tiles.
PFI retains the recipient's static terrain while permuting meteorological inputs;
terrain is not automatically added to the channel permutation list.

The local 8 GB RTX 4060 passed a synthetic float32 20->20, five-channel, batch-1
forward/backward probe with 32/64/3 parameters. This short GPU-only probe is not an
end-to-end speed, convergence or real-data accuracy acceptance result. Measure
memory and throughput for the selected model and task configuration.

See [task partitions](earth-task-splits.md), [metrics](earth-evaluation-metrics.md)
and [v1 compatibility](earth-3hourly-uploaded-model.md).

独立的三小时 **36×72（5°）** 训练发布、完整全球窗口与上传模型契约见 [5° 数据集](earth-merra2-5deg.md)；本文件的原 0.75° 契约继续兼容。
