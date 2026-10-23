# Uploaded Model Ls Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit independent `[batch, window]` Ls input to uploaded models without changing legacy uploaded models.

**Architecture:** A lightweight shared contract module will strictly normalize metadata, attach it to built models, validate Ls, and dispatch model calls. The runner will create aligned historical Ls windows; validator and inference paths will use the same dispatcher.

**Tech Stack:** Python 3, PyTorch, NumPy, NetCDF4, pytest.

---

## File Map

- Create `AstraAtmos_backend/backend/training_backbones/uploaded_model_contract.py`: metadata, validation, and dispatch.
- Modify `AstraAtmos_backend/backend/training_backbones/user_model_runner.py`: Ls windows and training loops.
- Modify `AstraAtmos_backend/backend/services/user_model_validator.py`: strict metadata and dry-run.
- Modify `AstraAtmos_backend/backend/services/inference_service.py`: uploaded prediction and analysis paths.
- Modify `AstraAtmos_backend/backend/tests/test_uploaded_model_runner.py`: core contract and dataset tests.
- Modify `AstraAtmos_backend/backend/tests/test_user_model_validator.py`: upload validation tests.
- Create `AstraAtmos_backend/backend/tests/test_uploaded_model_ls_inference.py`: inference propagation tests.
- Modify `AstraAtmos_backend/backend/tests/test_user_model_downloads.py`: documentation contract.
- Modify `docs/uploaded-model-training.md` and `docs/uploaded-model-template.py`: user guidance.

### Task 1: Shared Contract

- [ ] **Step 1: Write failing tests**

Add tests importing the desired API:

```python
class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.argument_count = 0

    def forward(self, x):
        self.argument_count = 1
        return x[:, :1, :1].repeat(1, 3, 1, 1, 1)


class RecordingLsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.received_ls = None

    def forward(self, x, ls):
        self.received_ls = ls
        return x[:, :1, :1].repeat(1, 3, 1, 1, 1)


x = torch.zeros(2, 3, 1, 8, 16)
ls = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
LS_MODEL_SPEC = {
    "name": "LsModel",
    "auxiliary_inputs": {
        "ls": {
            "required": True,
            "shape": ["batch", "window"],
            "dtype": "float32",
            "unit": "degree",
        }
    },
}

from training_backbones.uploaded_model_contract import (
    attach_uploaded_model_contract,
    normalize_auxiliary_inputs,
    run_uploaded_model,
)

def test_legacy_uploaded_model_receives_only_x():
    model = RecordingModel()
    attach_uploaded_model_contract(model, {"name": "Legacy"})
    run_uploaded_model(model, x, torch.ones(2, 3), context="legacy test")
    assert model.argument_count == 1

def test_ls_uploaded_model_receives_batch_window_tensor():
    model = RecordingLsModel()
    attach_uploaded_model_contract(model, LS_MODEL_SPEC)
    run_uploaded_model(model, x, ls, context="Ls test")
    assert torch.equal(model.received_ls, ls)
```

Parameterize rejection of unknown inputs, missing/extra Ls fields, `required=False`, wrong shape/dtype/unit, plus runtime missing, integer, wrong-rank, wrong-shape, NaN, and infinity Ls.

- [ ] **Step 2: Verify RED**

Run `python -m pytest AstraAtmos_backend/backend/tests/test_uploaded_model_runner.py -k "auxiliary or receives or ls_tensor" -q`.

Expected: collection fails because the contract module is absent.

- [ ] **Step 3: Implement minimal contract**

Create the module with these public functions:

```python
LS_AUXILIARY_INPUT = {
    "required": True,
    "shape": ["batch", "window"],
    "dtype": "float32",
    "unit": "degree",
}
CONTRACT_ATTRIBUTE = "_astraatmos_auxiliary_inputs"


def normalize_auxiliary_inputs(model_spec: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(model_spec, dict):
        raise ValueError("MODEL_SPEC must be a dict")
    if "auxiliary_inputs" not in model_spec:
        return {}
    auxiliary_inputs = model_spec["auxiliary_inputs"]
    if not isinstance(auxiliary_inputs, dict):
        raise ValueError("MODEL_SPEC.auxiliary_inputs must be a dict")
    if set(auxiliary_inputs) != {"ls"}:
        raise ValueError("MODEL_SPEC.auxiliary_inputs supports only ls")
    if auxiliary_inputs["ls"] != LS_AUXILIARY_INPUT:
        raise ValueError("MODEL_SPEC.auxiliary_inputs.ls must match the AstraAtmos Ls contract")
    return {"ls": dict(LS_AUXILIARY_INPUT)}


def attach_uploaded_model_contract(model, model_spec):
    setattr(model, CONTRACT_ATTRIBUTE, normalize_auxiliary_inputs(model_spec))
    return model


def uploaded_model_requires_ls(model) -> bool:
    return "ls" in getattr(model, CONTRACT_ATTRIBUTE, {})


def validate_ls_tensor(x, ls, context: str) -> torch.Tensor:
    if ls is None:
        raise ValueError(f"{context} requires Ls data")
    if not isinstance(ls, torch.Tensor):
        raise ValueError(f"{context} Ls must be a torch.Tensor")
    if not ls.is_floating_point():
        raise ValueError(f"{context} Ls must use a floating-point dtype, got {ls.dtype}")
    if ls.ndim != 2:
        raise ValueError(f"{context} Ls must be 2D, got {ls.ndim}D")
    expected = (int(x.shape[0]), int(x.shape[1]))
    if tuple(ls.shape) != expected:
        raise ValueError(f"{context} Ls shape mismatch: expected {expected}, got {tuple(ls.shape)}")
    if not torch.isfinite(ls).all():
        raise ValueError(f"{context} Ls must contain only finite values")
    return ls


def run_uploaded_model(model, x, ls=None, *, context="uploaded model"):
    if not uploaded_model_requires_ls(model):
        return model(x)
    return model(x, validate_ls_tensor(x, ls, context))
```

The only accepted declaration is `ls = {required: True, shape: [batch, window], dtype: float32, unit: degree}`. Dispatch calls `model(x)` for an absent declaration and `model(x, validated_ls)` for a valid declaration. Do not inspect signatures or retry after `TypeError`.

- [ ] **Step 4: Verify GREEN**

Repeat Step 2; expect all selected tests to pass.

### Task 2: Validator And Dry-Run

- [ ] **Step 1: Write failing validator tests**

Add `LS_MODEL_SOURCE` with `forward(self, x, ls)` and assertions for successful dry-run. Add failures for unknown auxiliary names and each malformed canonical field. Retain the legacy source test.

- [ ] **Step 2: Verify RED**

Run `python -m pytest AstraAtmos_backend/backend/tests/test_user_model_validator.py -q`.

Expected: the Ls model fails because dry-run passes only `x`.

- [ ] **Step 3: Implement shared validation and dispatch**

Normalize metadata before `build_model`, attach it after construction, and replace direct dry-run with:

```python
x = torch.zeros(2, 3, 1, 8, 16)
ls = torch.zeros(2, 3, dtype=torch.float32)
output = run_uploaded_model(model, x, ls, context="uploaded model dry-run")
```

The dispatcher ignores the locally constructed Ls for legacy models.

- [ ] **Step 4: Verify GREEN**

Repeat Step 2; expect all validator tests to pass.

### Task 3: Dataset Windows

- [ ] **Step 1: Write failing alignment tests**

Request `return_ls=True` from `prepare_tensors`, then assert a `window=3, horizon=2` dataset returns `ls.shape == [samples, 3]`, `ls[0] == [0, 1, 2]`, `x.shape[1] == 3`, and `y.shape[1] == 2`. Add a NetCDF fixture without Ls and assert the returned Ls is `None`, never synthetic.

- [ ] **Step 2: Verify RED**

Run `python -m pytest AstraAtmos_backend/backend/tests/test_uploaded_model_runner.py -k "prepare_tensors or window" -q`.

Expected: `return_ls` is unsupported.

- [ ] **Step 3: Implement optional timeline preservation**

Add `return_ls: bool = False` so the six-value legacy return remains intact. When requested, return `(x, y, ls, mean, std, height, width)`. Build `ls[i:i+window]` from the same index as `x`; never pass auxiliary Ls through `_clean_array`, so NaN/Inf reaches explicit validation. Allow genuinely absent Ls when no selected feature-loading path itself needs it.

- [ ] **Step 4: Verify GREEN**

Repeat Step 2; expect all alignment tests to pass.

### Task 4: Train, Validate, And Test

- [ ] **Step 1: Write failing loop tests**

Test `_split_train_test(x, y, ls)` produces three-tensor datasets and `_split_train_test(x, y, None)` preserves two-tensor datasets. Add a contract test ensuring the three runner phases use `run_uploaded_model`.

- [ ] **Step 2: Verify RED**

Run `python -m pytest AstraAtmos_backend/backend/tests/test_uploaded_model_runner.py -k "split_train or training_dispatch" -q`.

Expected: the split function cannot carry Ls and loops call models directly.

- [ ] **Step 3: Implement runner propagation**

Attach the normalized source contract inside `load_uploaded_model`. Load with `return_ls=True`, carry optional Ls through train/test datasets, unpack either `(x, y)` or `(x, ls, y)`, and route training, validation, and final metrics through `run_uploaded_model`. Validate required Ls before the first epoch. Keep optimizer, loss, early stopping, metrics, and saved weights unchanged.

- [ ] **Step 4: Verify GREEN**

Run `python -m pytest AstraAtmos_backend/backend/tests/test_uploaded_model_runner.py -q`; expect all tests to pass.

### Task 5: Prediction And Analysis

- [ ] **Step 1: Write failing inference tests**

Create focused tests using a tiny uploaded Ls model and monkeypatched tensors with `window=4, horizon=2`. Exercise `_predict_uploaded_task_window`, `_uploaded_task_test_set_arrays`, `_uploaded_task_permutation_importance`, and `_get_uploaded_model_test_results`; each must receive the exact Ls rows selected with `x`.

- [ ] **Step 2: Verify RED**

Run `python -m pytest AstraAtmos_backend/backend/tests/test_uploaded_model_ls_inference.py -q`.

Expected: Ls models fail because uploaded inference calls `model(x)`.

- [ ] **Step 3: Update every uploaded inference path**

Call `prepare_tensors` with its existing positional arguments plus `return_ls=True`, slice Ls with the same indices as `x`, transfer both to the same device, and call `run_uploaded_model` with path-specific context. Permutation importance shuffles only feature data; its corresponding Ls stays unchanged.

- [ ] **Step 4: Verify GREEN**

Run `python -m pytest AstraAtmos_backend/backend/tests/test_uploaded_model_ls_inference.py AstraAtmos_backend/backend/tests/test_trained_model_predict_contract.py AstraAtmos_backend/backend/tests/test_prediction_horizon_contract.py -q`; expect all tests to pass.

### Task 6: Documentation

- [ ] **Step 1: Write a failing download-contract test**

Assert the uploaded-model guide contains `auxiliary_inputs`, `forward(self, x, ls)`, and `[batch, window]`, while the default executable template remains a legacy `forward(self, x)` example.

- [ ] **Step 2: Verify RED**

Run `python -m pytest AstraAtmos_backend/backend/tests/test_user_model_downloads.py -q`; expect the new guide assertions to fail.

- [ ] **Step 3: Update guidance**

Document the exact opt-in metadata, shapes, dtype/unit, strict error rules, and compatibility behavior in `docs/uploaded-model-training.md`. Add a commented Ls example to `docs/uploaded-model-template.py` without changing the default executable model.

- [ ] **Step 4: Verify GREEN**

Repeat Step 2; expect all tests to pass.

### Task 7: Regression Verification

- [ ] **Step 1: Run focused suites**

Run:

```powershell
python -m pytest AstraAtmos_backend/backend/tests/test_uploaded_model_runner.py AstraAtmos_backend/backend/tests/test_user_model_validator.py AstraAtmos_backend/backend/tests/test_uploaded_model_ls_inference.py AstraAtmos_backend/backend/tests/test_user_model_downloads.py -q
```

Expected: all pass without new warnings.

- [ ] **Step 2: Run training and prediction regressions**

Run:

```powershell
python -m pytest AstraAtmos_backend/backend/tests/test_uploaded_training_contract.py AstraAtmos_backend/backend/tests/test_training_dataset_loader.py AstraAtmos_backend/backend/tests/test_trained_model_test_set_metrics.py AstraAtmos_backend/backend/tests/test_prediction_horizon_contract.py AstraAtmos_backend/backend/tests/test_trained_model_predict_contract.py -q
```

Expected: all pass.

- [ ] **Step 3: Run complete backend tests**

Run `python -m pytest AstraAtmos_backend/backend/tests -q` and report any environment-only failure with its exact error.

- [ ] **Step 4: Check scope**

Run `git diff --check` and `git status --short`. Expect no whitespace errors and preserve all pre-existing unrelated worktree changes.
