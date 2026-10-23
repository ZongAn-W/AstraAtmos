# Uploaded Model Training

Trusted lab users can upload a single Python file from the model training page and train it with the platform datasets.

The platform owns data loading, normalization, batching, the training loop, metrics, checkpoints, logs, and model testing. The uploaded file only defines the PyTorch architecture and a small parameter schema.

## Downloading and Renaming Uploaded Models

On the training page, expand **Manage models**, select a model, and choose
**Download source** or **Rename**. Renaming accepts a nonempty display name of
up to 120 characters, trims surrounding whitespace, and updates that package's
name in the list and current model summary. Save commits the change; Cancel or
Escape closes the editor. Revalidation preserves the edited name.

Both actions require authentication and are available only to the uploader:

| Action | API | Result |
| --- | --- | --- |
| Download source | `GET /api/user-models/{model_id}/download` | Original uploaded bytes as a `.py` attachment, retaining the original basename |
| Rename | `PATCH /api/user-models/{model_id}` with `{"display_name":"New name"}` | Updated model metadata |

Other accounts receive 403; nonexistent or deleted packages receive 404.
Downloads also return 404 when the stored source is missing. Invalid names are
rejected with 400 or 422. Invalid or pending models may still be renamed or
downloaded so their source can be repaired locally.

Renaming changes metadata only: the package ID, version, source, content hash,
validation result, and model identity already fixed in training tasks remain
unchanged. The name inside `MODEL_SPEC` is not rewritten, and re-uploading the
downloaded file follows the normal upload/version rules. Matching display names
do not merge packages. Downloads contain architecture source; trained weights
remain available from training task artifacts.

## Required Exports

Every upload must export:

- `MODEL_SPEC`: metadata and adjustable parameters.
- `build_model(config)`: a callable that returns a `torch.nn.Module`.
- One or more `torch.nn.Module` classes used by `build_model`.

Use [uploaded-model-template.py](uploaded-model-template.py) as the starting point.

## Tensor Contract

The model receives tensors with this shape:

```text
[batch, window, channels, height, width]
```

It must return predictions with this shape:

```text
[batch, horizon, 1, height, width]
```

The platform passes these core config keys to `build_model(config)`:

- `in_channels`
- `window`
- `horizon`
- `height`
- `width`
- `selected_channels`

Any custom fields declared in `MODEL_SPEC["parameters"]` are also included in `config`.

## Mars Input Channels and Checkpoints

For Mars, `selected_channels` contains only user-selected auxiliary inputs, such as
`["U", "D"]`. O3 is the fixed prediction target and the first input channel. The
complete model input order is `["O3", *selected_channels]`: selecting no auxiliaries
still supplies one O3 channel, and selecting U supplies `["O3", "U"]`.

Both the official `demo3.py` runner and uploaded `user_model_runner.py` runner save
the shared `astraatmos_mars_forecast_checkpoint_v1` contract. The checkpoint retains
auxiliary-only `training_contract.selected_channels`, while
`normalization.input_channel_order` records the complete order. `input_mean`,
`input_scale`, and `constant_channel_mask` each have one entry per input channel;
each mean/scale entry has the training grid shape. Loading rejects reordered or
missing channel names, incorrect counts or shapes, non-finite statistics, and
non-positive input scales. Providing invalid saved normalization never silently
refits it.

Inference applies input statistics in that exact order, and independently uses
`target_mean` and `target_scale` to restore predicted O3 values (with the existing
`1e-6` scale epsilon). A constant target can retain `target_scale=0`; input scales
must remain positive. Scaled-volume caches include checkpoint statistics so two
tasks with the same data and channels cannot reuse each other's normalization.

Older versioned checkpoints that recorded only auxiliary channel names are
rejected. Retrain them, or migrate them separately only after verifying the
original tensor order and all statistics; there is no automatic metadata repair.
Legacy bare `state_dict` weights retain the existing normalization-refit path.
These conventions apply to Mars; the Earth contract remains independent.

Mars checkpoints also bind the initial training data identity. `mcd_overview`
records the original full MCD directory; `openmars_mcd` records both OpenMARS and
MCD directories, with source-tagged manifests. The runner retains the initial
statistics and file identity instead of reloading them after training. Both
prediction paths verify the saved directories, file names, sizes and modification
times before cache/window/model access. A changed source returns HTTP 409
`dataset_version_changed` and requires retraining. Complete identities are marked
`verified`; historical weights without file identity stay explicitly `legacy`.
Incomplete versioned checkpoints are rejected. See the
[Mars dataset identity contract](dataset-registry.md#mars-数据绑定与预测校验).

## Declaring Which Dataset Feeds a Model

Active Earth training uses `earth_merra2_3hourly_v1` and its independent
[v1 tile](earth-3hourly-uploaded-model.md) or
[v2 full-grid](earth-3hourly-fullgrid-model.md) contract. Task creation reuses the
compatibility report saved by upload or manual revalidation, verifies ownership,
valid status, concrete dataset/schema evidence, source hash, parameter schema and
batch range, then freezes the source/schema/parameters and queues the task. It no
longer runs another dry-run for the task's actual windows, channels, batch or
custom parameters. Runtime output/gradient checks and checkpoint strict reload
remain required. The daily `earth_merra2` declaration and examples below document
the historical, retired daily feed.

By default an uploaded model is **Mars-only**. To train it on Earth MERRA-2 data it must
opt in explicitly:

```python
MODEL_SPEC = {
    "name": "EarthConvBaseline",
    "datasets": {
        "earth_merra2": {
            "grid": [[36, 72]],
            "window": [7],
            "horizon": [3],
            "auxiliary_inputs": [],
            "target_leading_channels": 1,
        }
    },
    "parameters": {"hidden_dim": {"type": "int", "default": 16, "min": 4, "max": 64}},
}
```

Being valid for Mars is **not** enough for Earth. A model is accepted for Earth training
only when all of these hold:

1. the source file is present and hashes to the recorded digest;
2. it passes the shared safety gate (imports limited to `torch` / `numpy`, no
   `open` / `eval` / `exec` / `compile` / `__import__`, no `system` / `popen` / `Popen` / `run`);
3. `MODEL_SPEC["datasets"]` declares the `earth_merra2` feed;
4. for the active three-hour Earth feed, the declaration uses the independent
   v1 tile or v2 full-grid schema; `ls` and Earth `topography` are declared via
   `derived_inputs`, never through Mars `auxiliary_inputs`;
5. a real dry-run builds the model with the selected three-hour window,
   horizon, grid and declared derived inputs. The complete active contract is
   documented in [Earth three-hour uploaded models](earth-3hourly-uploaded-model.md).

Any failure is reported with a reason; the training page disables "start experiment"
until the verdict is positive. An upload with no `datasets` block keeps its historical
Mars behaviour and reports `"compatible": false` for Earth with the reason
`MODEL_SPEC does not declare the earth_merra2 dataset feed`.

The retired daily Earth example below does not describe the active three-hour
feed. Active Earth models receive `TO3` plus selected meteorological channels,
and may additionally declare UTC-derived `ls` or the server-owned ETOPO
`topography` tensor. Mars MOLA declarations remain Mars-only. See the active
[v1 contract](earth-3hourly-uploaded-model.md) and [Earth terrain asset](earth-topography-asset.md).

Ask the server for the verdict instead of guessing:

```text
GET /api/user-models/{model_id}/earth-compatibility
```

```json
{
  "package_id": "…", "display_name": "EarthConvBaseline", "version": 3,
  "validation_status": "valid", "source_available": true,
  "compatible": true, "reasons": [], "warnings": [],
  "datasets": {"earth_merra2": {"grid": [[36, 72]], "window": [7], "horizon": [3]}},
  "output_shape": [1, 3, 1, 36, 72]
}
```

The same verdict is embedded in every upload / revalidate response as
`validation_report.earth` (with `validation_report.datasets`), so it also appears in
`GET /api/user-models`.

## Training on Earth With an Uploaded Model

Start the task with `model_source=uploaded` and a top-level `uploaded_model_id`;
validated parameter values go under `hyperparameters.custom_model_params`:

```json
{
  "model_script": "unified_training.py",
  "model_name": "Earth uploaded conv",
  "model_source": "uploaded",
  "uploaded_model_id": "<UserModelPackage.id>",
  "dataset_id": "earth_merra2_daily_v2",
  "hyperparameters": {
    "training_dataset": "earth_merra2_daily_v2",
    "model_source": "uploaded",
    "selected_channels": ["U10M", "T2M"],
    "epochs": 5,
    "custom_model_params": {"hidden_dim": 16, "dropout": 0.0}
  }
}
```

The server pins the package id, version, content hash, validated parameter values and the
hash-verified source text at task creation:

- uploading a newer version later does **not** change what this task trains;
- the checkpoint embeds the verified source, so the model is rebuilt and predicted after a
  service restart, and still works if the original file is deleted or tampered with;
- no pickle/unserialize of user objects happens anywhere: the embedded text is re-checked
  by the same AST gate and `build_model(config)` is called; weights load with
  `torch.load(..., weights_only=True)`;
- when the original file is missing or changed, prediction succeeds from the embedded copy
  and says so in the response `warnings` field and the prediction page.

Parameter values are checked against the uploaded package's own
`MODEL_SPEC["parameters"]` schema (type, `min`/`max`, `select` options, unknown keys). A
model that reads a declared parameter still builds even if the stored schema is incomplete,
because the declaration's defaults are used as a fallback.

## Optional Ls Input

Models that use historical solar longitude can opt into a separate `ls` tensor. The declaration must exactly match this contract:

```python
MODEL_SPEC = {
    "name": "ExamplePhaseModel",
    "description": "Model with explicit solar-longitude context.",
    "auxiliary_inputs": {
        "ls": {
            "required": True,
            "shape": ["batch", "window"],
            "dtype": "float32",
            "unit": "degree",
        }
    },
    "parameters": {},
}
```

The model then accepts two inputs:

```python
def forward(self, x, ls):
    # x:  [batch, window, channels, height, width]
    # ls: [batch, window]
    ...
```

Each Ls value is measured in degrees and corresponds to the historical frame at the same window index. Ls always covers the input window, even when `window` and `horizon` differ.

Models that omit `auxiliary_inputs` remain single-input models and receive only `model(x)`. The platform does not inspect the function signature, insert Ls into a feature channel, or retry calls after `TypeError`.

For declared Ls models, the platform rejects missing data, non-floating tensors, values with the wrong `[batch, window]` shape, and any `NaN` or infinite value. It never substitutes zeros, random values, or `None` for required Ls data. Unknown auxiliary inputs and deviations from the metadata block above are rejected during upload validation.

## Optional MOLA Topography Input

Models can opt into static Mars surface elevation independently of Ls:

```python
MODEL_SPEC = {
    "name": "ExampleTopographyModel",
    "description": "Model with explicit MOLA terrain context.",
    "auxiliary_inputs": {
        "topography": {
            "required": True,
            "shape": ["batch", 1, "height", "width"],
            "dtype": "float32",
            "unit": "meter",
        }
    },
    "parameters": {},
}
```

The corresponding method is:

```python
def forward(self, x, topography):
    # x:           [batch, window, channels, height, width]
    # topography:  [batch, 1, height, width], float32 meters
    ...
```

Topography is static. It has no window dimension, is not repeated through time,
and is not included in `in_channels`. The platform reads the actual latitude
and longitude coordinates of the selected training dataset and periodically
resamples its built-in NASA/PDS MOLA grid to those coordinates. Matching array
dimensions alone are not treated as spatial alignment.

A model can declare both auxiliary inputs:

```python
MODEL_SPEC = {
    "name": "ExampleLsTopographyModel",
    "auxiliary_inputs": {
        "ls": {
            "required": True,
            "shape": ["batch", "window"],
            "dtype": "float32",
            "unit": "degree",
        },
        "topography": {
            "required": True,
            "shape": ["batch", 1, "height", "width"],
            "dtype": "float32",
            "unit": "meter",
        },
    },
    "parameters": {},
}

def forward(self, x, ls, topography):
    ...
```

Dispatch is based only on `MODEL_SPEC` and always uses this order: `x`, then
declared `ls`, then declared `topography`. The platform does not inspect the
function signature, catch `TypeError` to retry, or pass undeclared inputs.

Upload dry-run, training, validation, final metrics, test evaluation,
permutation importance, and formal prediction all use the same dispatcher.
Permutation importance shuffles only ordinary `x` features; Ls and topography
remain fixed. The static `[1, height, width]` terrain tensor is expanded to
`[batch, 1, height, width]` without copying it for each sample.

A declared topography model fails clearly if the MOLA asset is absent, corrupt,
non-finite, or cannot align to the target coordinates. There is no zero, random,
`None`, or flat-terrain fallback. Models without a topography declaration do
not load the asset. See [mola-topography-asset.md](mola-topography-asset.md) for
the source product and reproduction process.

## Parameter Schema

Supported parameter types:

- `int`: requires `default`, `min`, and `max`.
- `float`: requires `default`, `min`, and `max`.
- `bool`: requires `default`.
- `select`: requires string `default` and non-empty string `options`.

Example:

```python
MODEL_SPEC = {
    "name": "ExampleUploadedModel",
    "description": "Small convolutional baseline.",
    "parameters": {
        "hidden_dim": {"type": "int", "default": 16, "min": 4, "max": 128},
        "dropout": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.9},
        "use_bias": {"type": "bool", "default": True},
        "activation": {"type": "select", "default": "relu", "options": ["relu", "gelu"]},
    },
}
```

## Validation Rules

Upload and manual revalidation use a bounded child process,
starting with a default import/declaration budget of 120 seconds. Set the positive integer
`USER_MODEL_VALIDATION_TIMEOUT_SECONDS` in the backend environment and restart the
backend to change it. All execution checks remain required. A timeout terminates
the child and records `validation_report.code="uploaded_model_validation_timeout"`;
the package stays invalid until revalidation succeeds. Earth compatibility reports
`status="unknown"`, `compatible=false` and the same code, and the UI displays
"Validation timed out. Please retry." Existing reports containing only the legacy
timeout error text receive the same treatment. This budget is separate from the
30-second checkpoint completion/reload gate.

After the isolated child imports the source and validates its dataset declaration
and parameter schema, a full-grid v2 declaration extends the total budget to the
larger of the ordinary setting and `USER_MODEL_FULL_GRID_VALIDATION_TIMEOUT_SECONDS`
(default 900 seconds). The deadline always counts from process start; the parent
never executes uploaded source to choose it. A hung import or invalid declaration
does not receive the extension. Full-grid upload/revalidation still uses all 16
channel combinations, the full 240x480 grid, default model parameters, finite
gradient checks and eval sample independence. Upload and revalidation wait in a
background thread so unrelated API requests remain responsive.

Explicit `earth_probe` calls use the same larger budget for full-grid v2 and still
check the requested configuration. Mars and v1 tile checks retain the ordinary
budget; explicit validator timeouts remain authoritative. Task creation reuses
the saved report and does not run these probes. The process result is drained
before joining its sender to avoid false timeouts for large reports. The process
must still exit successfully within the deadline. See
[full-grid admission](earth-3hourly-fullgrid-model.md).

Version 1 accepts these import roots:

- `torch`
- `numpy`

The validator rejects filesystem, subprocess, dynamic execution, and network-style escape hatches such as `open`, `eval`, `exec`, `compile`, `__import__`, `system`, `popen`, `Popen`, and `run`.

Before a model can be trained, the platform:

1. Parses the file as UTF-8 Python.
2. Checks imports and disallowed calls.
3. Imports the module in a validation process.
4. Normalizes `MODEL_SPEC["parameters"]`, optional `MODEL_SPEC["datasets"]` and optional auxiliary-input metadata.
5. Calls `build_model(config)`.
6. Runs a Mars dry forward pass with x shape `[2, 3, 1, 8, 16]` and, when declared, Ls shape `[2, 3]`, requiring output shape `[2, 3, 1, 8, 16]`.
7. When the model declares `earth_merra2`, additionally builds it for Earth and runs `[1, 7, C, 36, 72]` → `[1, 3, 1, 36, 72]` for `C` in `{1, 5}`, recording the result as `validation_report.earth`.

Step 7 runs the same AST safety gate as steps 1–2. An uploaded model that fails it is
still uploadable as a Mars model; it simply never becomes Earth-capable.

## Minimal Model

```python
from torch import nn


MODEL_SPEC = {
    "name": "RepeatLastFrame",
    "parameters": {},
}


class RepeatLastFrame(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon

    def forward(self, x):
        last_frame = x[:, -1, :1]
        return last_frame.unsqueeze(1).repeat(1, self.horizon, 1, 1, 1)


def build_model(config):
    return RepeatLastFrame(config["horizon"])
```

## Earth task partitions

The active three-hour Earth feed shares `earth_raw_utc_timeline_v1` with the official DLinear. Positive train/validation/test ratios (default 0.7/0.2/0.1) allocate the entire raw UTC timeline before any windows are built. Normalization fits only the task train interval; uploaded backtests must keep their full input and target inside one task partition. Release files, manifest splits and fingerprint remain unchanged. Frozen task boundaries survive queue recovery and strict checkpoint reload; older three-hour artifacts retain manifest semantics. Comparison requires the same test time range and window definition. See [Earth task partitions](earth-task-splits.md).
