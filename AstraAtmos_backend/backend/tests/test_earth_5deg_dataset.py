"""Conservative spatial aggregation and a real low-resolution training round trip."""
import json
import hashlib
import asyncio
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest
import torch

from scripts.build_earth_merra2_5deg import build_dataset, overlap_weights, regrid
from services.earth_dataset import CHANNELS, validate_earth_3hourly_dataset, EarthThreeHourlyWindows
from services.earth_dataset_metadata import file_sha256, read_earth_3hourly_release
from services.dataset_registry import DatasetRegistry
from services.earth_training_contract import build_earth_training_spec
from services.earth_task_split import build_earth_task_split
from services.earth_training_artifact import load_earth_training_artifact
from services.earth_prediction_service import build_prediction_context, run_earth_prediction
from models.training_scripts.earth_daily import run_training
from test_earth_3hourly_data_contract import make_threehour_release, write_manifest
from test_earth_3hourly_training_service import service_environment

LOWRES = 'earth_merra2_3hourly_5deg_v1'


def test_download_template_passes_process_isolation(tmp_path, monkeypatch):
    from services.user_model_validator import UserModelValidator
    monkeypatch.setenv('OMP_NUM_THREADS', '2')
    template = Path(__file__).resolve().parents[3] / 'docs/earth-3hourly-5deg-model-template.py'
    model = tmp_path / 'model.py'; model.write_bytes(template.read_bytes())
    validation = UserModelValidator().validate_file(model)
    assert validation.ok, validation.errors
    proof = validation.report_dict()['earth_datasets'][LOWRES]
    assert proof['status'] == 'available' and proof['output_shape'] == [2, 1, 1, 36, 72]


def test_overlap_preserves_area_mean_and_propagates_missing():
    source_lat = np.column_stack((np.arange(240) * .75 - 90, (np.arange(240) + 1) * .75 - 90))
    source_lon = np.column_stack((np.arange(480) * .75 - 180, (np.arange(480) + 1) * .75 - 180))
    target_lat = np.column_stack((np.arange(36) * 5 - 90, (np.arange(36) + 1) * 5 - 90))
    target_lon = np.column_stack((np.arange(72) * 5 - 180, (np.arange(72) + 1) * 5 - 180))
    lat = overlap_weights(source_lat, target_lat, spherical=True)
    lon = overlap_weights(source_lon, target_lon)
    values = np.random.default_rng(12).normal(size=(2, 240, 480)).astype('float32')
    result = regrid(values, np.ones_like(values, dtype='uint8'), lat, lon)
    source_area = np.diff(np.sin(np.deg2rad(source_lat)), axis=1)[:, 0]
    target_area = np.diff(np.sin(np.deg2rad(target_lat)), axis=1)[:, 0]
    np.testing.assert_allclose(np.average(result.mean(axis=2), weights=target_area, axis=1),
        np.average(values.mean(axis=2), weights=source_area, axis=1), rtol=0, atol=1e-8)
    # Independent direct overlap computation for a cell crossing native boundaries.
    a, b = target_lat[9]; c, d = target_lon[12]
    lw = np.maximum(0, np.sin(np.deg2rad(np.minimum(b, source_lat[:, 1]))) - np.sin(np.deg2rad(np.maximum(a, source_lat[:, 0]))))
    ow = np.maximum(0, np.minimum(d, source_lon[:, 1]) - np.maximum(c, source_lon[:, 0]))
    expected = np.sum(values[0] * lw[:, None] * ow[None, :]) / (lw.sum() * ow.sum())
    assert result[0, 9, 12] == pytest.approx(expected, abs=1e-7)
    mask = np.ones_like(values, dtype='uint8'); mask[0, 0, 0] = 0
    missing = regrid(values, mask, lat, lon)
    assert np.isnan(missing[0, 0, 0]) and np.isfinite(missing[1]).all()


@pytest.fixture(scope='module')
def lowres_release(tmp_path_factory):
    root = tmp_path_factory.mktemp('earth5deg')
    source = make_threehour_release(root / 'source', days=3, missing=False)
    # Real varying fields make train-only statistics and metrics observable.
    with netCDF4.Dataset(str(source / 'earth_merra2_3hourly.nc'), 'a') as ds:
        for index, channel in enumerate(CHANNELS):
            ds[channel][:] = 200 + index + np.arange(24, dtype='float32')[:, None, None] + np.linspace(0, 1, 240, dtype='float32')[None, :, None]
    import xarray as xr
    with xr.open_dataset(source / 'earth_merra2_3hourly.nc', engine='netcdf4', mask_and_scale=False) as ds:
        stats = validate_earth_3hourly_dataset(ds)
    manifest = json.loads((source / 'manifest.json').read_text())
    manifest.update(variables=stats, data_bytes=(source / 'earth_merra2_3hourly.nc').stat().st_size,
                    data_sha256=file_sha256(source / 'earth_merra2_3hourly.nc'))
    write_manifest(source, manifest)
    result = build_dataset(source, root / 'lowres', progress=None)
    return result, root


def test_lowres_publication_preserves_time_units_and_identity(lowres_release):
    release, root = lowres_release
    assert release.metadata['dataset_id'] == LOWRES
    assert release.metadata['grid_shape'] == [36, 72]
    assert len(release.dates) == 24
    assert release.metadata['time']['start'] == '2020-01-01T01:30:00Z'
    assert all(v['missing_rate'] == 0 for v in release.metadata['variables'])
    cached = read_earth_3hourly_release(root / 'lowres', verification_cache_dir=root / 'proof')
    reused = read_earth_3hourly_release(root / 'lowres', verification_cache_dir=root / 'proof')
    assert reused.verification_cached and reused.metadata == cached.metadata
    assert build_dataset(root / 'source', root / 'lowres', progress=None).metadata['dataset_fingerprint'] == release.metadata['dataset_fingerprint']
    wrong_registry = DatasetRegistry(root / 'daily', earth_3hourly_package_dir=root / 'lowres')
    assert wrong_registry.get_dataset('earth_merra2_3hourly_v1')['availability'] == 'invalid'


def test_lowres_task_admission_and_queue_restore_bind_the_new_release(lowres_release, service_environment):
    release, root = lowres_release
    module, _ = service_environment
    registry = DatasetRegistry(root / 'missing_daily', earth_3hourly_5deg_package_dir=root / 'lowres')
    service = module.TrainingService(); service._scheduler_started = True
    task = asyncio.run(service.start_training(user_id=1, custom_model_name='Earth 5 degree queue', model_script='ignored.py',
        hyperparameters={'epochs': 1, 'window': 2, 'horizon': 1, 'selected_channels': [],
                         'train_ratio': .5, 'validation_ratio': .25, 'test_ratio': .25},
        dataset_id=LOWRES, dataset_registry=registry))
    assert task.dataset_id == LOWRES and task.status == 'queued'
    execution = asyncio.run(service._prepare_training_execution(task, dataset_registry=registry))
    spec = execution['earth_training_spec']
    assert spec['dataset_binding']['dataset_id'] == LOWRES
    assert spec['training_profile']['grid_shape'] == [36, 72]
    assert spec['data_path'] == str(release.data_path)
    assert spec['metrics_contract']['aggregation'] == 'forecast_origin_lead_cell_area_weighted'


@pytest.mark.parametrize('uploaded', [False, True])
def test_lowres_training_checkpoint_prediction_and_diagnostics(lowres_release, uploaded, monkeypatch):
    release, root = lowres_release
    registry = DatasetRegistry(root / 'missing_daily', earth_3hourly_5deg_package_dir=root / 'lowres')
    binding = registry.build_training_binding(LOWRES)
    ratios = {'train_ratio': .5, 'validation_ratio': .25, 'test_ratio': .25}
    split = build_earth_task_split(release.dates, 2, 1, ratios)
    reference = None
    terrain = None
    source = 'uploaded' if uploaded else 'official'
    if uploaded:
        from test_earth_topography import asset
        from services.user_model_validator import UserModelValidator
        from services.earth_derived_inputs import freeze_topography_for_reference
        from services.user_model_service import UserModelService
        terrain_root = root / 'terrain'; terrain_root.mkdir(exist_ok=True)
        terrain_asset, _ = asset(terrain_root, rows=36, columns=72)
        monkeypatch.setattr('services.earth_topography.EARTH_TOPOGRAPHY_5DEG_PATH', terrain_root / 'terrain.nc')
        spec_literal = {'name': 'Earth5degTerrain', 'parameters': {},
            'datasets': {LOWRES: {**__import__('training_backbones.earth_3hourly_uploaded_contract', fromlist=['LOWRES_FEED']).LOWRES_FEED,
                                 'window': [2, 56], 'horizon': [1, 24]}},
            'derived_inputs': {'ls': {'required': True, 'shape': ['batch', 'window'], 'dtype': 'float32', 'unit': 'degree'},
                'topography': {'required': True, 'shape': ['batch', 1, 'height', 'width'], 'dtype': 'float32', 'unit': 'meter'}}}
        code = ('import torch\nfrom torch import nn\nMODEL_SPEC = ' + repr(spec_literal) + '''
class Forecast(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.9))
        self.horizon = config['horizon']
    def forward(self, x, ls, topography):
        y = x[:, -1:, :1] * self.scale + topography[:, None] / 10000 + ls[:, -1:, None, None, None] / 360
        return y.expand(-1, self.horizon, -1, -1, -1).contiguous()
def build_model(config):
    return Forecast(config)
''')
        model_file = root / 'model.py'; model_file.write_bytes(code.encode('utf-8'))
        result = UserModelValidator(timeout_seconds=None).validate_file(model_file)
        assert result.ok, result.errors
        package = SimpleNamespace(id='5deg-model', storage_path=str(model_file), content_hash=hashlib.sha256(code.encode()).hexdigest(),
            validation_status='valid', validation_report=json.dumps(result.report_dict()), display_name='5deg', version=1)
        verdict = UserModelService.earth_compatibility_for_package(package, dataset_id=LOWRES)
        assert verdict['compatible'], json.dumps(verdict)
        assert not UserModelService.earth_compatibility_for_package(package, dataset_id='earth_merra2_3hourly_v1')['compatible']
        reference = {'package_id': package.id, 'version': 1, 'content_hash': package.content_hash,
            'source_text': code, 'source_path': str(model_file), 'display_name': '5deg',
            'param_schema': {}, 'custom_model_params': {}, 'contract_schema': 'astraatmos_earth_3hourly_5deg_model_v1'}
        terrain = freeze_topography_for_reference(reference, release)
    spec = build_earth_training_spec(task_id=901, dataset_binding=binding, task_split=split,
        uploaded_model=reference, topography=terrain,
        hyperparameters={**ratios, 'epochs': 1, 'batch_size': 2, 'window': 2, 'horizon': 1,
                         'selected_channels': ['U10M', 'T2M'], 'model_source': source})
    output = root / ('uploaded901.pth' if uploaded else 'task901.pth')
    previous = torch.get_num_threads(); torch.set_num_threads(2)
    try:
        result = run_training(spec, output, registry, device='cpu', cache_root=root / 'cache')
        checkpoint = load_earth_training_artifact(output, expected_binding=binding, expected_task_id=901)
        assert checkpoint.model_config['height'] == 36 and checkpoint.model_config['width'] == 72
        assert checkpoint.training_contract['grid_shape'] == [36, 72]
        assert checkpoint.run['spatial_grid_shape'] == [36, 72]
        assert checkpoint.normalization['fit_step_count'] == 12
        task_hypers = {**spec['hyperparameters'], '_earth_task_split': split}
        if terrain is not None:
            task_hypers['_earth_topography'] = terrain
        task = SimpleNamespace(id=901, status='completed', output_model_path=str(output),
            model_source=source, hyperparameters=json.dumps(task_hypers), **binding)
        context = build_prediction_context(task, registry)
        assert context.grid_shape == [36, 72]
        prediction = run_earth_prediction(task, context.origins['start'], registry, device='cpu')
        assert prediction['dataset_id'] == LOWRES and prediction['grid']['shape'] == [36, 72]
        assert len(prediction['target_timestamps']) == 1
        from services.earth_diagnostics import compute_earth_diagnostics
        diagnostic = compute_earth_diagnostics(task, registry, sample_windows=2, scatter_points=16,
            pfi_repeats=1, device='cpu')
        assert diagnostic['dataset_id'] == LOWRES
    finally:
        torch.set_num_threads(previous)
