import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from services.user_model_service import UserModelService
from services.user_model_validator import UserModelValidator
from training_backbones.earth_3hourly_uploaded_contract import (
    DATASET_ID, CONTRACT_SCHEMA, FEED, FULL_GRID_CONTRACT_SCHEMA, FULL_GRID_FEED,
    build_config, contract_schema_from_source, contract_schema_for_reference,
    contract_profile, forward, prepare_model, validate_eval_batch_independence,
)
from training_backbones.uploaded_model_dataset_spec import normalize_dataset_declarations, DatasetCapabilityError
from training_backbones.uploaded_model_earth_gate import evaluate_package_earth_compatibility


SOURCE = '''
import torch
from torch import nn
MODEL_SPEC = SPEC
class FullGrid(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.horizon = config['horizon']
    def forward(self, inputs):
        if tuple(inputs.shape[-2:]) != (240,480):
            raise ValueError('The model requires one full global map')
        return inputs[:,-1:,:1].expand(-1,self.horizon,-1,-1,-1) + self.bias
def build_model(config):
    return FullGrid(config)
'''


@pytest.fixture(scope='module', autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def full_package(tmp_path):
    declaration = copy.deepcopy(FULL_GRID_FEED)
    declaration.update(window=[2, 20], horizon=[1, 20])
    spec = {'name': 'FullGrid', 'parameters': {}, 'datasets': {DATASET_ID: declaration}}
    source = SOURCE.replace('= SPEC', '= ' + repr(spec))
    path = tmp_path / 'fullgrid.py'
    path.write_text(source, encoding='utf-8')
    result = UserModelValidator(timeout_seconds=None).validate_file(path)
    assert result.ok, result.errors
    return SimpleNamespace(id='full-grid-model', display_name='FullGrid', version=1,
        storage_path=str(path), content_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
        validation_status='valid', validation_report=json.dumps(result.report_dict())), result


def test_full_grid_upload_and_cached_compatibility_preserve_spatial_contract(full_package, tmp_path):
    package, result = full_package
    verdict = result.earth_compatibilities[DATASET_ID]
    assert verdict['contract_schema'] == FULL_GRID_CONTRACT_SCHEMA
    assert verdict['output_shape'] == [2, 1, 1, 240, 480]
    assert verdict['validation_scope'] == 'declared_channels_probe_windows'
    service = UserModelService(storage_root=tmp_path / 'storage')
    async def get_package(*args):
        return package
    service.get_package_for_user = get_package
    cached = asyncio.run(service.get_earth_compatibility(package.id, 1, dataset_id=DATASET_ID))
    assert cached['compatible'] is True and cached['status'] == 'available'
    assert cached['default_batch_size'] == 1 and cached['max_batch_size'] == 64
    assert cached['spatial_input_shape'] == [240, 480]


@pytest.mark.parametrize("schema", [CONTRACT_SCHEMA, FULL_GRID_CONTRACT_SCHEMA])
def test_upload_budget_signal_uses_the_normalized_executed_contract(tmp_path, schema):
    feed = copy.deepcopy(FULL_GRID_FEED if schema == FULL_GRID_CONTRACT_SCHEMA else FEED)
    feed.update(window=[2], horizon=[1])
    spec = {"name": "Budget", "parameters": {}, "datasets": {DATASET_ID: feed}}
    source = SOURCE.replace("= SPEC", "= " + repr(spec))
    if schema == CONTRACT_SCHEMA:
        source = source.replace("(240,480)", "(24,48)")
    # Composed declarations must work without executing source in the parent.
    source = source.replace("MODEL_SPEC = ", "MODEL_SPEC = dict(", 1)
    source = source.replace("\nclass FullGrid", ")\nclass FullGrid", 1)
    path = tmp_path / "model.py"
    path.write_text(source, encoding="utf-8")
    announced = []
    result = UserModelValidator._validate_file_in_process(
        path, on_contract_validated=announced.append,
    )
    assert result.ok, result.errors
    assert announced == [schema]
    # Invalid full-grid declarations cannot request the extended budget.
    invalid_path = tmp_path / "invalid.py"
    invalid_path.write_text(source.replace("'frequency_hours': 3", "'frequency_hours': 6"), encoding="utf-8")
    announced.clear()
    result = UserModelValidator._validate_file_in_process(invalid_path, on_contract_validated=announced.append)
    assert not result.ok and announced == []


def test_full_grid_phase_message_and_large_report_survive_spawn(tmp_path):
    feed = {**copy.deepcopy(FULL_GRID_FEED), "window": [2], "horizon": [1]}
    spec = {"name": "LargeFullGrid", "parameters": {}, "datasets": {DATASET_ID: feed},
            "description": "full-grid report " * 4096}
    path = tmp_path / "model.py"
    path.write_text(SOURCE.replace("= SPEC", "= " + repr(spec)), encoding="utf-8")
    result = UserModelValidator().validate_file(path)
    assert result.ok, result.errors
    assert result.description == spec["description"]
    assert result.earth_compatibilities[DATASET_ID]["output_shape"] == [2, 1, 1, 240, 480]


def test_full_grid_task_probe_checks_actual_window_and_horizon(full_package):
    package, _ = full_package
    probe = {'input_channel_order': ['TO3'], 'window': 20, 'horizon': 20,
             'batch_size': 1, 'custom_model_params': {}}
    verdict = evaluate_package_earth_compatibility(package, UserModelValidator(timeout_seconds=None),
                                                   dataset_id=DATASET_ID, earth_probe=probe)
    assert verdict.compatible, verdict.reasons
    assert verdict.output_shape == [2, 20, 1, 240, 480]
    probe['batch_size'] = 65
    verdict = evaluate_package_earth_compatibility(package, UserModelValidator(timeout_seconds=None),
                                                   dataset_id=DATASET_ID, earth_probe=probe)
    assert not verdict.compatible and 'batch size' in verdict.reasons[0]


def test_task_probe_rejects_frozen_contract_mismatch(full_package):
    package, _ = full_package
    verdict = evaluate_package_earth_compatibility(
        package, UserModelValidator(timeout_seconds=None), dataset_id=DATASET_ID,
        earth_probe={'input_channel_order': ['TO3'], 'window': 2, 'horizon': 1,
                     'batch_size': 1, 'custom_model_params': {}, 'contract_schema': CONTRACT_SCHEMA},
    )
    assert not verdict.compatible
    assert verdict.code == 'uploaded_model_contract_invalid'
    assert 'Frozen spatial contract' in verdict.reasons[0]


def test_task_resolution_does_not_execute_a_configuration_probe(full_package):
    from services.training_service import TrainingService
    package, _ = full_package
    package.param_schema = '{}'
    class Packages:
        async def get_package_for_user(self, *args):
            return package
    class UnexpectedValidator:
        def validate_file(self, *args, **kwargs):
            pytest.fail('task resolution must use the stored upload verdict')
    service = TrainingService(earth_model_validator=UnexpectedValidator())
    reference, _ = asyncio.run(service._resolve_earth_uploaded_model(
        user_id=1, uploaded_model_id=package.id, user_model_service=Packages(),
        custom_model_params={}, dataset_id=DATASET_ID, input_channel_order=['TO3'],
        window=20, horizon=20, batch_size=1,
    ))
    assert reference.contract_schema == FULL_GRID_CONTRACT_SCHEMA
    assert reference.content_hash == package.content_hash
    assert reference.source_text == Path(package.storage_path).read_bytes().decode('utf-8')


@pytest.mark.parametrize('mutation', [
    {'spatial_tile_shape': [24,48]}, {'schema': 'unknown'}, {'tensor_layout': 'BTHWC'},
])
def test_full_grid_schema_cannot_be_combined_with_a_tile_contract(mutation):
    feed = {**copy.deepcopy(FULL_GRID_FEED), **mutation}
    with pytest.raises(DatasetCapabilityError):
        normalize_dataset_declarations({'datasets': {DATASET_ID: feed}})


def test_contract_schema_is_read_without_executing_source():
    spec = {'datasets': {DATASET_ID: {'schema': FULL_GRID_CONTRACT_SCHEMA}}}
    assert contract_schema_from_source('MODEL_SPEC = ' + repr(spec) + '\nraise RuntimeError()') == FULL_GRID_CONTRACT_SCHEMA
    for source in ('MODEL_SPEC = build_spec()', 'MODEL_SPEC = {}',
                   'MODEL_SPEC = ' + repr(spec) + '\nMODEL_SPEC = ' + repr(spec)):
        with pytest.raises(ValueError):
            contract_schema_from_source(source)


def test_full_grid_factory_metadata_cannot_silently_use_tiles():
    config = build_config(['TO3'], window=20, horizon=20, contract_schema=FULL_GRID_CONTRACT_SCHEMA)
    assert (config['height'], config['width']) == (240, 480)
    assert config['spatial_tile_shape'] == [240,480]
    assert contract_profile(CONTRACT_SCHEMA)['shape'] == (24,48)
    model = prepare_model(torch.nn.Identity(), window=20, contract_schema=FULL_GRID_CONTRACT_SCHEMA)
    with pytest.raises(ValueError, match='240,480'):
        forward(model, torch.zeros(1,20,1,24,48), horizon=20)


@pytest.mark.parametrize('field,value', [('contract_schema', CONTRACT_SCHEMA), ('output_shape', [2,1,1,24,48]),
    ('validation_scope', None), ('batch_size', 8)])
def test_incomplete_full_grid_evidence_cannot_be_available(full_package, tmp_path, field, value):
    package, result = full_package
    report = result.report_dict()
    report['earth_datasets'][DATASET_ID][field] = value
    package.validation_report = json.dumps(report)
    service = UserModelService(storage_root=tmp_path / 'storage')
    async def get_package(*args):
        return package
    service.get_package_for_user = get_package
    cached = asyncio.run(service.get_earth_compatibility(package.id, 1, dataset_id=DATASET_ID))
    assert not cached['compatible'] and cached['status'] == 'unknown'


@pytest.mark.parametrize('schema,batch_size,expected_batches', [
    (FULL_GRID_CONTRACT_SCHEMA, 1, {1, 2}),
    (FULL_GRID_CONTRACT_SCHEMA, 2, {1, 2}),
    (FULL_GRID_CONTRACT_SCHEMA, 4, {1, 2, 4}),
    (CONTRACT_SCHEMA, 8, {1, 2, 3, 8}),
])
def test_independence_probes_only_supported_full_grid_batches(schema, batch_size, expected_batches):
    batches = set()
    class Bounded(torch.nn.Module):
        def forward(self, inputs):
            batches.add(inputs.shape[0])
            if schema == FULL_GRID_CONTRACT_SCHEMA and inputs.shape[0] > max(2, batch_size):
                raise ValueError('The probe exceeded the requested batch size')
            return inputs[:, -1:, :1]
    model = prepare_model(Bounded(), window=1, contract_schema=schema)
    validate_eval_batch_independence(model, window=1, channels=1, horizon=1, batch_size=batch_size)
    assert batches == expected_batches


@pytest.mark.parametrize('reject_training_batch', [False, True])
def test_full_grid_task_probe_checks_the_requested_training_batch(tmp_path, reject_training_batch):
    feed = copy.deepcopy(FULL_GRID_FEED)
    feed.update(window=[2], horizon=[1])
    spec = {'name': 'RequestedBatch', 'parameters': {}, 'datasets': {DATASET_ID: feed}}
    source = SOURCE.replace('MODEL_SPEC = SPEC', 'MODEL_SPEC = ' + repr(spec))
    if reject_training_batch:
        source = source.replace('    def forward(self, inputs):\n',
            '    def forward(self, inputs):\n'
            '        if self.training and inputs.shape[0] > 2:\n'
            "            raise ValueError('Requested batch is unsupported during training')\n")
    path = tmp_path / 'requested_batch.py'
    path.write_text(source, encoding='utf-8')
    probe = {'input_channel_order': ['TO3'], 'window': 2, 'horizon': 1,
             'batch_size': 4, 'custom_model_params': {}}
    result = UserModelValidator(timeout_seconds=None).validate_file(path, earth_probe=probe)
    verdict = result.earth_compatibilities[DATASET_ID]
    assert verdict['compatible'] is (not reject_training_batch)
    assert verdict['batch_size'] == 4
    if reject_training_batch:
        assert 'unsupported during training' in verdict['errors'][0]
    else:
        assert verdict['output_shape'] == [2, 1, 1, 240, 480]


@pytest.mark.parametrize('schema', [CONTRACT_SCHEMA, FULL_GRID_CONTRACT_SCHEMA])
@pytest.mark.parametrize('declaration_style', ['constant', 'composed_dict'])
def test_executed_schema_is_frozen_for_nonliteral_model_declarations(tmp_path, schema, declaration_style):
    from services import training_service
    from services.earth_model_source import build_uploaded_earth_model
    from services.earth_training_contract import build_earth_training_spec
    from test_earth_3hourly_training_service import ServerRegistry, binding

    feed = copy.deepcopy(FULL_GRID_FEED if schema == FULL_GRID_CONTRACT_SCHEMA else FEED)
    feed.update(window=[2], horizon=[1])
    spec = {'name': 'DynamicDeclaration', 'parameters': {}, 'datasets': {DATASET_ID: feed}}
    if declaration_style == 'constant':
        declaration = 'SCHEMA = ' + repr(schema) + '\nMODEL_SPEC = ' + repr(spec).replace(repr(schema), 'SCHEMA')
    else:
        declaration = ('FEED = ' + repr(feed)
                       + '\nBASE = ' + repr({'name': spec['name'], 'parameters': {}})
                       + '\nMODEL_SPEC = dict(BASE, datasets={' + repr(DATASET_ID) + ': FEED})')
    shape = contract_profile(schema)['shape']
    source = SOURCE.replace('MODEL_SPEC = SPEC', declaration).replace('(240,480)', repr(shape))
    path = tmp_path / 'dynamic_model.py'
    path.write_text(source, encoding='utf-8')
    validation = UserModelValidator(timeout_seconds=None).validate_file(path)
    assert validation.ok, validation.errors
    with pytest.raises(ValueError):
        contract_schema_from_source(source)
    package = SimpleNamespace(id='dynamic', display_name=spec['name'], version=1,
        storage_path=str(path), content_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
        validation_status='valid', validation_report=json.dumps(validation.report_dict()), param_schema='{}')
    class Packages:
        async def get_package_for_user(self, *args):
            return package
    class UnexpectedValidator:
        def validate_file(self, *args, **kwargs):
            pytest.fail('dynamic declarations must retain the schema from upload validation')
    service = training_service.TrainingService()
    service._earth_model_validator = UnexpectedValidator()
    reference, _ = asyncio.run(service._resolve_earth_uploaded_model(user_id=1, uploaded_model_id=package.id,
        user_model_service=Packages(), custom_model_params={}, dataset_id=DATASET_ID,
        input_channel_order=['TO3'], window=2, horizon=1, batch_size=1))
    assert reference.contract_schema == schema
    frozen = reference.checkpoint_reference()
    assert contract_schema_for_reference(frozen) == schema
    internal = build_earth_training_spec(task_id=991, dataset_binding={'dataset_id': DATASET_ID},
        hyperparameters={'model_source': 'uploaded', 'window': 2, 'horizon': 1, 'batch_size': 1},
        uploaded_model=frozen)
    assert internal['uploaded_model']['contract_schema'] == schema
    model, config, _ = build_uploaded_earth_model(reference=frozen, input_channel_order=['TO3'],
        window=2, horizon=1, height=240, width=480, dataset_id=DATASET_ID, contract_schema=schema)
    assert config['contract_schema'] == schema
    assert tuple(forward(model, torch.zeros(1, 2, 1, *shape), horizon=1).shape) == (1, 1, 1, *shape)
    raw_hypers = {**internal['hyperparameters'], '_uploaded_model_id': package.id,
                  '_uploaded_model_content_hash': package.content_hash,
                  '_earth_uploaded_reference': frozen, '_earth_uploaded_contract_schema': schema}
    task = SimpleNamespace(id=991, **binding(), uploaded_model_id=package.id, uploaded_model_version=1,
                           hyperparameters=json.dumps(raw_hypers))
    restored = service._restore_3hourly_training_spec(task, dataset_registry=ServerRegistry(tmp_path / 'verified.nc'))
    assert restored['uploaded_model']['contract_schema'] == schema
    if schema == CONTRACT_SCHEMA:
        legacy = {key: value for key, value in frozen.items() if key != 'contract_schema'}
        assert contract_schema_for_reference(legacy) == CONTRACT_SCHEMA
        raw_hypers.pop('_earth_uploaded_contract_schema')
        raw_hypers['_earth_uploaded_reference'] = legacy
        task.hyperparameters = json.dumps(raw_hypers)
        restored = service._restore_3hourly_training_spec(task, dataset_registry=ServerRegistry(tmp_path / 'verified.nc'))
        assert restored['uploaded_model']['contract_schema'] == CONTRACT_SCHEMA


def test_frozen_schema_still_rejects_literal_source_disagreement():
    from services.dataset_identity import DatasetRequestError
    from services.earth_training_contract import build_earth_training_spec
    source = 'MODEL_SPEC = ' + repr({'datasets': {DATASET_ID: {'schema': FULL_GRID_CONTRACT_SCHEMA}}})
    reference = {'package_id': 'mismatch', 'version': 1, 'content_hash': 'a' * 64,
                 'source_text': source, 'param_schema': {}, 'contract_schema': CONTRACT_SCHEMA}
    with pytest.raises(ValueError, match='disagrees'):
        contract_schema_for_reference(reference)
    with pytest.raises(DatasetRequestError, match='disagrees') as error:
        build_earth_training_spec(task_id=991, dataset_binding={'dataset_id': DATASET_ID},
            hyperparameters={'model_source': 'uploaded', 'batch_size': 1}, uploaded_model=reference)
    assert error.value.code == 'uploaded_model_contract_invalid'
