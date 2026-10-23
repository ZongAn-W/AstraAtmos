import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services import user_model_validator  # noqa: E402
from services.user_model_validator import (  # noqa: E402
    UserModelValidator, UserModelValidationResult, VALIDATION_TIMEOUT_CODE,
)


VALID_MODEL_SOURCE = """
import torch
from torch import nn

MODEL_SPEC = {
    "name": "TinyModel",
    "description": "Tiny repeat baseline for validator tests.",
    "parameters": {
        "hidden_dim": {
            "type": "int",
            "default": 8,
            "min": 1,
            "max": 32,
        },
        "dropout": {
            "type": "float",
            "default": 0.1,
            "min": 0.0,
            "max": 0.5,
        },
        "use_bias": {
            "type": "bool",
            "default": True,
        },
        "activation": {
            "type": "select",
            "default": "relu",
            "options": ["relu", "gelu"],
        },
    },
}


class TinyModel(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon

    def forward(self, x):
        last = x[:, -1, :1]
        return last.unsqueeze(1).repeat(1, self.horizon, 1, 1, 1)


def build_model(config):
    return TinyModel(config["horizon"])
"""

LS_MODEL_SOURCE = """
import torch
from torch import nn

MODEL_SPEC = {
    "name": "TinyLsModel",
    "description": "Tiny model requiring historical solar longitude.",
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


class TinyLsModel(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon

    def forward(self, x, ls):
        if ls.shape != x.shape[:2]:
            raise ValueError(f"misaligned Ls: {ls.shape} vs {x.shape[:2]}")
        if ls.dtype != torch.float32:
            raise ValueError(f"wrong Ls dtype: {ls.dtype}")
        last = x[:, -1, :1]
        return last.unsqueeze(1).repeat(1, self.horizon, 1, 1, 1)


def build_model(config):
    return TinyLsModel(config["horizon"])
"""

TOPOGRAPHY_MODEL_SOURCE = """
import torch
from torch import nn

MODEL_SPEC = {
    "name": "TinyTopographyModel",
    "description": "Tiny model requiring MOLA terrain.",
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

class TinyTopographyModel(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon

    def forward(self, x, topography):
        if topography.shape != (x.shape[0], 1, x.shape[-2], x.shape[-1]):
            raise ValueError(f"wrong topography shape: {topography.shape}")
        if topography.dtype != torch.float32 or not torch.isfinite(topography).all():
            raise ValueError("invalid topography dtype or values")
        if torch.all(topography == topography.flatten()[0]):
            raise ValueError("topography must be real non-flat MOLA terrain")
        return x[:, -1:, :1].repeat(1, self.horizon, 1, 1, 1)

def build_model(config):
    return TinyTopographyModel(config["horizon"])
"""

LS_TOPOGRAPHY_MODEL_SOURCE = TOPOGRAPHY_MODEL_SOURCE.replace(
    '"topography": {',
    '"ls": {\n'
    '            "required": True,\n'
    '            "shape": ["batch", "window"],\n'
    '            "dtype": "float32",\n'
    '            "unit": "degree",\n'
    '        },\n'
    '        "topography": {',
    1,
).replace(
    "def forward(self, x, topography):",
    "def forward(self, x, ls, topography):\n"
    "        if ls.shape != x.shape[:2]:\n"
    "            raise ValueError(f'wrong Ls shape: {ls.shape}')",
    1,
)


def _write_temp_model(temp_dir: str, source: str) -> Path:
    path = Path(temp_dir) / "uploaded_model.py"
    path.write_text(source, encoding="utf-8")
    return path


def _validate_source(source: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = _write_temp_model(temp_dir, source)
        return UserModelValidator.validate_file(path)


def _validate_source_with_validator(source: str, validator: UserModelValidator):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = _write_temp_model(temp_dir, source)
        return validator.validate_file(path)


def test_valid_model_passes_and_reports_metadata():
    result = _validate_source(VALID_MODEL_SOURCE)

    assert result.ok is True
    assert result.errors == []
    assert result.display_name == "TinyModel"
    assert result.description == "Tiny repeat baseline for validator tests."
    assert result.param_schema["hidden_dim"]["default"] == 8
    assert result.output_shape == [2, 3, 1, 8, 16]


def test_ls_model_dry_run_receives_declared_auxiliary_input():
    result = _validate_source(LS_MODEL_SOURCE)

    assert result.ok is True
    assert result.errors == []
    assert result.output_shape == [2, 3, 1, 8, 16]


def test_topography_model_dry_run_receives_real_mola_tensor():
    result = _validate_source(TOPOGRAPHY_MODEL_SOURCE)

    assert result.ok is True
    assert result.errors == []
    assert result.output_shape == [2, 3, 1, 8, 16]


def test_ls_topography_model_dry_run_receives_fixed_argument_order():
    result = _validate_source(LS_TOPOGRAPHY_MODEL_SOURCE)

    assert result.ok is True
    assert result.errors == []


def test_topography_model_dry_run_fails_when_mola_asset_is_missing(
    monkeypatch,
    tmp_path,
):
    missing = tmp_path / "missing-mola.nc"
    monkeypatch.setattr(user_model_validator, "MOLA_TOPOGRAPHY_PATH", missing)

    result = _validate_source_with_validator(
        TOPOGRAPHY_MODEL_SOURCE,
        UserModelValidator(timeout_seconds=None),
    )

    assert result.ok is False
    assert any(str(missing) in error and "(8, 16)" in error for error in result.errors)


def test_legacy_and_ls_dry_runs_do_not_load_mola(monkeypatch):
    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("MOLA must not load for undeclared topography")

    monkeypatch.setattr(
        user_model_validator,
        "prepare_topography_grid",
        fail_if_loaded,
        raising=False,
    )

    legacy_result = _validate_source_with_validator(
        VALID_MODEL_SOURCE,
        UserModelValidator(timeout_seconds=None),
    )
    ls_result = _validate_source_with_validator(
        LS_MODEL_SOURCE,
        UserModelValidator(timeout_seconds=None),
    )

    assert legacy_result.ok is True
    assert ls_result.ok is True


def test_unknown_auxiliary_input_is_rejected():
    result = _validate_source(
        LS_MODEL_SOURCE.replace('"ls": {', '"season": {', 1)
    )

    assert result.ok is False
    assert any("auxiliary" in error.lower() and "only ls" in error for error in result.errors)


def test_malformed_ls_auxiliary_input_is_rejected_before_dry_run():
    result = _validate_source(
        LS_MODEL_SOURCE.replace('"unit": "degree"', '"unit": "radian"', 1)
    )

    assert result.ok is False
    assert any("auxiliary_inputs.ls" in error for error in result.errors)


def test_disallowed_import_fails_before_import():
    result = _validate_source(
        VALID_MODEL_SOURCE.replace("import torch", "import os\nimport torch", 1)
    )

    assert result.ok is False
    assert any("Disallowed import: os" in error for error in result.errors)


def test_missing_model_spec_fails():
    result = _validate_source(VALID_MODEL_SOURCE.replace("MODEL_SPEC =", "MISSING_SPEC =", 1))

    assert result.ok is False
    assert any("MODEL_SPEC" in error for error in result.errors)


def test_bad_output_shape_fails():
    result = _validate_source(
        VALID_MODEL_SOURCE.replace(
            "return last.unsqueeze(1).repeat(1, self.horizon, 1, 1, 1)",
            "return last",
            1,
        )
    )

    assert result.ok is False
    assert any("output shape" in error for error in result.errors)


def test_invalid_numeric_param_schema_fails():
    result = _validate_source(
        VALID_MODEL_SOURCE.replace(
            '"min": 1,\n            "max": 32,',
            '"min": 64,\n            "max": 32,',
            1,
        )
    )

    assert result.ok is False
    assert any("hidden_dim" in error for error in result.errors)


def test_invalid_parameter_name_fails():
    result = _validate_source(
        VALID_MODEL_SOURCE.replace('"hidden_dim": {', '"bad-name": {', 1)
    )

    assert result.ok is False
    assert any(
        "Invalid parameter name" in error or "bad-name" in error
        for error in result.errors
    )


def test_keyword_parameter_name_fails():
    result = _validate_source(
        VALID_MODEL_SOURCE.replace('"hidden_dim": {', '"class": {', 1)
    )

    assert result.ok is False
    assert any(
        "Invalid parameter name" in error or "class" in error
        for error in result.errors
    )


def test_select_options_must_be_strings():
    result = _validate_source(
        VALID_MODEL_SOURCE.replace(
            '"options": ["relu", "gelu"]',
            '"options": ["relu", 3]',
            1,
        )
    )

    assert result.ok is False
    assert any(
        "activation" in error or "select" in error or "options" in error
        for error in result.errors
    )


def test_select_options_must_be_non_empty_strings():
    result = _validate_source(
        VALID_MODEL_SOURCE.replace(
            '"default": "relu",\n            "options": ["relu", "gelu"]',
            '"default": "",\n            "options": [""]',
            1,
        )
    )

    assert result.ok is False
    assert any(
        "activation" in error or "select" in error or "options" in error
        for error in result.errors
    )


def test_forward_timeout_fails():
    previous_children = {child.pid for child in multiprocessing.active_children()}
    result = _validate_source_with_validator(
        VALID_MODEL_SOURCE.replace(
            "return last.unsqueeze(1).repeat(1, self.horizon, 1, 1, 1)",
            "while True:\n            pass",
            1,
        ),
        UserModelValidator(timeout_seconds=0.5),
    )

    assert result.ok is False
    assert result.code == VALIDATION_TIMEOUT_CODE
    assert result.report_dict()["code"] == VALIDATION_TIMEOUT_CODE
    assert {child.pid for child in multiprocessing.active_children()} <= previous_children
    assert any(
        "timeout" in error.lower() or "timed out" in error.lower()
        for error in result.errors
    )


@pytest.mark.parametrize("configured,expected", [(None, 120), ("240", 240)])
def test_validation_budget_configuration_reaches_class_and_instance(configured, expected):
    environment = os.environ.copy()
    environment.pop("USER_MODEL_VALIDATION_TIMEOUT_SECONDS", None)
    if configured is not None:
        environment["USER_MODEL_VALIDATION_TIMEOUT_SECONDS"] = configured
    probe = '''
import json
from pathlib import Path
import dotenv
dotenv.load_dotenv = lambda *args, **kwargs: None
import config
from services.user_model_validator import UserModelValidator, UserModelValidationResult
budgets = []
def validate(path, timeout_seconds, **kwargs):
    budgets.append(timeout_seconds)
    return UserModelValidationResult(ok=True)
UserModelValidator._validate_file_with_timeout = staticmethod(validate)
UserModelValidator.validate_file(Path("model.py"))
UserModelValidator().validate_file(Path("model.py"))
print(json.dumps({"configured": config.USER_MODEL_VALIDATION_TIMEOUT_SECONDS, "budgets": budgets}))
'''
    completed = subprocess.run([sys.executable, "-c", probe], cwd=BACKEND_DIR, env=environment,
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"configured": expected, "budgets": [expected, expected]}


def test_full_grid_probe_uses_the_dedicated_validation_budget(monkeypatch):
    calls = []
    result = UserModelValidationResult(ok=True)

    def isolated(path, budget, **kwargs):
        calls.append(budget)
        return result

    monkeypatch.setattr(UserModelValidator, "_validate_file_with_timeout", staticmethod(isolated))
    monkeypatch.setattr(user_model_validator.config, "USER_MODEL_FULL_GRID_VALIDATION_TIMEOUT_SECONDS", 300)
    validator = UserModelValidator()
    assert validator.validate_file(
        Path("model.py"),
        earth_probe={"contract_schema": "astraatmos_earth_3hourly_fullgrid_model_v2"},
    ) is result
    assert calls == [300]


@pytest.mark.parametrize('explicit', [0.5, 4, 120, None])
def test_full_grid_probe_preserves_explicit_timeout(monkeypatch, explicit):
    calls = []
    result = UserModelValidationResult(ok=True)
    def isolated(path, budget, **kwargs):
        calls.append(budget)
        return result
    def in_process(path, **kwargs):
        calls.append(None)
        return result
    monkeypatch.setattr(UserModelValidator, '_validate_file_with_timeout', staticmethod(isolated))
    monkeypatch.setattr(UserModelValidator, '_validate_file_in_process', staticmethod(in_process))
    validator = UserModelValidator(timeout_seconds=explicit)
    assert validator.validate_file(Path('model.py'), earth_probe={
        'contract_schema': 'astraatmos_earth_3hourly_fullgrid_model_v2',
    }) is result
    assert calls == [explicit]


@pytest.mark.parametrize('configured,expected', [(None, 900), ('420', 420)])
def test_full_grid_budget_configuration_reaches_default_probe(configured, expected):
    environment = os.environ.copy()
    environment.pop('USER_MODEL_FULL_GRID_VALIDATION_TIMEOUT_SECONDS', None)
    environment['USER_MODEL_VALIDATION_TIMEOUT_SECONDS'] = '120'
    if configured is not None:
        environment['USER_MODEL_FULL_GRID_VALIDATION_TIMEOUT_SECONDS'] = configured
    script = '''
import dotenv
dotenv.load_dotenv = lambda *args, **kwargs: None
from pathlib import Path
import json
from services.user_model_validator import UserModelValidator, UserModelValidationResult
budgets = []
def isolated(path, budget, **kwargs):
    budgets.append(budget)
    return UserModelValidationResult(ok=True)
UserModelValidator._validate_file_with_timeout = staticmethod(isolated)
probe = {'contract_schema': 'astraatmos_earth_3hourly_fullgrid_model_v2'}
UserModelValidator().validate_file(Path('model.py'), earth_probe=probe)
UserModelValidator.validate_file(Path('model.py'), earth_probe=probe)
UserModelValidator().validate_file(Path('model.py'), earth_probe={'contract_schema': 'astraatmos_earth_3hourly_uploaded_model_v1'})
print(json.dumps(budgets))
'''
    completed = subprocess.run([sys.executable, '-c', script], cwd=BACKEND_DIR, env=environment,
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == [expected, expected, 120]


@pytest.mark.parametrize('configured', ['0', '-1', 'invalid'])
def test_full_grid_budget_rejects_invalid_configuration(configured):
    environment = {**os.environ, 'USER_MODEL_FULL_GRID_VALIDATION_TIMEOUT_SECONDS': configured}
    script = 'import dotenv; dotenv.load_dotenv = lambda *args, **kwargs: None; import config'
    completed = subprocess.run([sys.executable, '-c', script], cwd=BACKEND_DIR, env=environment,
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode != 0
    assert 'ValueError' in completed.stderr


def test_large_validation_report_is_drained_before_child_exit(tmp_path):
    previous_children = {child.pid for child in multiprocessing.active_children()}
    description = 'validation report ' * 4096
    source = VALID_MODEL_SOURCE.replace('Tiny repeat baseline for validator tests.', description)
    path = tmp_path / 'large_report.py'
    path.write_text(source, encoding='utf-8')
    result = UserModelValidator(timeout_seconds=20).validate_file(path)
    assert result.ok, result.errors
    assert result.description == description
    assert {child.pid for child in multiprocessing.active_children()} <= previous_children


@pytest.mark.parametrize("full_grid_budget,delay,expected_ok", [
    (10, 5, True), (None, 5, False), (10, 11, False),
])
def test_full_grid_contract_signal_extends_only_the_default_absolute_deadline(
    monkeypatch, full_grid_budget, delay, expected_ok,
):
    from types import SimpleNamespace
    from queue import Empty

    elapsed = [0.0]
    class ResultQueue:
        def __init__(self):
            self.messages = [
                {"phase": user_model_validator._FULL_GRID_VALIDATION_PHASE},
                {"ok": True},
            ]
        def get(self, **kwargs):
            message = self.messages[0]
            if "phase" not in message and elapsed[0] + kwargs["timeout"] < delay:
                elapsed[0] += kwargs["timeout"]
                raise Empty
            self.messages.pop(0)
            elapsed[0] = 1.0 if "phase" in message else delay
            return message
        def close(self):
            pass
    class Process:
        def __init__(self):
            self.exitcode = None
            self.alive = True
        def start(self):
            pass
        def join(self, timeout):
            # The child exits only once its simulated computation is finished.
            if elapsed[0] + timeout >= delay:
                elapsed[0] = max(elapsed[0], delay)
                self.alive = False
                self.exitcode = 0
        def is_alive(self):
            return self.alive
        def terminate(self):
            self.alive = False
            self.exitcode = -1
        def kill(self):
            pytest.fail("The fake child should terminate immediately")
    queue = ResultQueue()
    process = Process()
    monkeypatch.setattr(user_model_validator.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(multiprocessing, "get_context", lambda kind: SimpleNamespace(
        Queue=lambda **kwargs: queue, Process=lambda **kwargs: process,
    ))
    result = UserModelValidator._validate_file_with_timeout(
        Path("model.py"), 2, full_grid_timeout_seconds=full_grid_budget,
    )
    assert result.ok is expected_ok
    if not expected_ok:
        assert result.code == VALIDATION_TIMEOUT_CODE


@pytest.mark.parametrize('exitcode', [1, None])
def test_successful_payload_requires_successful_child_exit(monkeypatch, exitcode):
    from types import SimpleNamespace
    calls = []
    class ResultQueue:
        def get(self, **kwargs):
            calls.append('get')
            return {'ok': True}
        def close(self):
            calls.append('close')
    class Process:
        def __init__(self):
            self.exitcode = exitcode
            self.alive = exitcode is None
        def start(self):
            calls.append('start')
        def join(self, timeout):
            calls.append('join')
        def is_alive(self):
            return self.alive
        def terminate(self):
            calls.append('terminate')
            self.alive = False
        def kill(self):
            pytest.fail('Terminated fake process is already stopped')
    monkeypatch.setattr(multiprocessing, 'get_context', lambda kind: SimpleNamespace(
        Queue=lambda **kwargs: ResultQueue(), Process=lambda **kwargs: Process(),
    ))
    result = UserModelValidator(timeout_seconds=.5).validate_file(Path('model.py'))
    assert not result.ok
    assert calls.index('get') < calls.index('join')
    assert calls[-1] == 'close'
    if exitcode is None:
        assert result.code == VALIDATION_TIMEOUT_CODE
        assert 'terminate' in calls
    else:
        assert 'without a result: 1' in result.errors[0]


@pytest.mark.parametrize("configured", ["0", "-1", "invalid"])
def test_validation_budget_configuration_rejects_invalid_values(configured):
    environment = {**os.environ, "USER_MODEL_VALIDATION_TIMEOUT_SECONDS": configured}
    probe = "import dotenv; dotenv.load_dotenv = lambda *args, **kwargs: None; import config"
    completed = subprocess.run([sys.executable, "-c", probe], cwd=BACKEND_DIR, env=environment,
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode != 0
    assert "ValueError" in completed.stderr


def test_explicit_budget_and_in_process_override_are_retained(monkeypatch):
    calls = []
    result = UserModelValidationResult(ok=True)
    def isolated(path, budget, **kwargs):
        calls.append((path, budget))
        return result
    def in_process(path, **kwargs):
        calls.append((path, None))
        return result
    monkeypatch.setattr(UserModelValidator, "_validate_file_with_timeout", staticmethod(isolated))
    monkeypatch.setattr(UserModelValidator, "_validate_file_in_process", staticmethod(in_process))
    path = Path("model.py")
    assert UserModelValidator(timeout_seconds=0.5).validate_file(path) is result
    assert UserModelValidator(timeout_seconds=None).validate_file(path) is result
    assert calls == [(path, 0.5), (path, None)]


def test_timeout_code_survives_validation_payload_and_report():
    result = UserModelValidationResult(ok=False, errors=["budget exceeded"], code=VALIDATION_TIMEOUT_CODE)
    payload = user_model_validator._validation_payload(result)
    assert payload["code"] == VALIDATION_TIMEOUT_CODE
    assert UserModelValidationResult(**payload).report_dict()["code"] == VALIDATION_TIMEOUT_CODE
    assert "code" not in UserModelValidationResult(ok=True).report_dict()


if __name__ == "__main__":
    test_valid_model_passes_and_reports_metadata()
    test_disallowed_import_fails_before_import()
    test_missing_model_spec_fails()
    test_bad_output_shape_fails()
    test_invalid_numeric_param_schema_fails()
    test_invalid_parameter_name_fails()
    test_keyword_parameter_name_fails()
    test_select_options_must_be_strings()
    test_select_options_must_be_non_empty_strings()
    test_forward_timeout_fails()
    print("user model validator tests passed")
