"""Global checkpoint verification receives a budget independent of the old 30s cap."""
from types import SimpleNamespace

import pytest

import config
from services import earth_training_artifact as artifact
from training_backbones.earth_3hourly_uploaded_contract import (
    CONTRACT_SCHEMA, FULL_GRID_CONTRACT_SCHEMA, LOWRES_CONTRACT_SCHEMA,
)


@pytest.mark.parametrize('schema,source,override,expected', [
    (CONTRACT_SCHEMA, 'uploaded', None, 30.),
    (FULL_GRID_CONTRACT_SCHEMA, 'uploaded', None, 901.),
    (LOWRES_CONTRACT_SCHEMA, 'uploaded', None, 901.),
    (None, 'official', None, 30.),
    (LOWRES_CONTRACT_SCHEMA, 'uploaded', .01, .01),
])
def test_global_default_budget_and_explicit_deadline(schema, source, override, expected, monkeypatch):
    joins = []
    metadata_reads = []
    class Pipe:
        def close(self):
            pass
        def poll(self):
            return True
        def recv(self):
            return {'ok': True}
    class Process:
        pid = 1
        def start(self):
            pass
        def join(self, timeout):
            joins.append(timeout)
        def is_alive(self):
            return False
        def close(self):
            pass
    def read(path):
        metadata_reads.append(path)
        return SimpleNamespace(model_source=source, model_config={'contract_schema':schema})
    monkeypatch.setattr(config, 'USER_MODEL_VALIDATION_TIMEOUT_SECONDS', 121)
    monkeypatch.setattr(config, 'USER_MODEL_FULL_GRID_VALIDATION_TIMEOUT_SECONDS', 901)
    monkeypatch.setattr(artifact, 'load_earth_training_artifact', read)
    monkeypatch.setattr(artifact.multiprocessing, 'get_context', lambda _: SimpleNamespace(
        Pipe=lambda **kwargs: (Pipe(), Pipe()), Process=lambda **kwargs: Process()))
    artifact.verify_earth_model_reload_isolated('saved.pth', timeout_seconds=override)
    assert joins == [expected]
    assert metadata_reads == ([] if override is not None else ['saved.pth'])
