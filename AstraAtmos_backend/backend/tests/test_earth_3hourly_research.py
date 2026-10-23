"""Streaming UTC daily aggregation and compact annual analyses on real NetCDF."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.earth_analysis import router
from services.earth_overview_service import EarthOverviewError
from services.earth_research_service import EarthResearchService, THREE_HOUR_DAILY_AGGREGATION
from test_earth_3hourly_training_runner import synthetic_training_release

ID = 'earth_merra2_3hourly_v1'


@pytest.fixture(scope='module')
def research(synthetic_training_release):
    source = synthetic_training_release
    metadata = copy.deepcopy(source.metadata)
    metadata['grid'].update(latitude_range=[-90., 90.], longitude_range=[-180., 180.],
                            cell_bounds={'latitude': [-90., 90.], 'longitude': [-180., 180.]},
                            latitude_step=.75, longitude_step=.75, coverage='global', wrap_longitude=True)
    metadata['time']['calendar'] = 'proleptic_gregorian'
    metadata['manifest'] = {'variables': {entry['id']: {'min': 100. + index, 'max': 103. + index}
                                           for index, entry in enumerate(metadata['variables'])}}
    release = replace(source, metadata=metadata)
    registry = SimpleNamespace(get_earth_overview_snapshot=lambda dataset, fingerprint: release)
    return EarthResearchService(registry), registry, release


def test_context_describes_source_threehour_time_not_daily(research):
    service, _, release = research
    result = service.get_context(ID, release.metadata['dataset_fingerprint'])
    assert result['time']['kind'] == 'iso-datetime'
    assert result['time']['start'] == '2020-01-01T01:30:00Z'
    assert result['time']['end'] == '2020-01-30T22:30:00Z'
    assert result['time']['count'] == 240 and result['time']['step_unit'] == 'hour'
    assert result['geometry']['shape'] == [240, 480]
    assert result['source_meta']['cadence'] == '3-hour mean (UTC)'
    assert result['source_meta']['analysis_aggregation'] == THREE_HOUR_DAILY_AGGREGATION
    assert result['unavailable']['diurnal'] == 'three_hourly_diurnal_analysis_not_implemented'


def test_annual_daily_average_reads_eight_frames_and_keeps_only_aggregates(research, monkeypatch):
    import services.earth_3hourly_overview as reads
    service, _, release = research
    original = reads._read_values
    sizes = []
    def bounded(*args, **kwargs):
        sizes.append(args[4] - args[3])
        return original(*args, **kwargs)
    monkeypatch.setattr(reads, '_read_values', bounded)
    data = service._variable_data(release, 2020, 'TO3')
    assert max(sizes) == 8
    assert data['global'].shape == (30,)
    assert data['zonal'].shape == (30, 240)
    assert data['annual'].shape == (240, 480)
    # Native values are 100 + step*.01 + row*.001 + col*.0005.
    expected = np.float32(100.035) + np.arange(240)[:, None] * .001 + np.arange(480)[None, :] * .0005
    weights = np.diff(np.sin(np.deg2rad(np.linspace(-90., 90., 241))))
    expected_global = np.einsum('ij,i->', expected, weights) / (480 * weights.sum())
    assert data['global'][0] == pytest.approx(expected_global, abs=1e-5)
    np.testing.assert_allclose(data['annual'][0, :4], 101.195 + np.arange(4) * .0005, atol=1e-5)
    before = len(sizes)
    assert service._variable_data(release, 2020, 'TO3') is data
    assert len(sizes) == before


def test_research_and_spatial_http_are_daily_aggregates_with_compact_grids(research):
    service, _, release = research
    app = FastAPI()
    app.state.earth_research_service = service
    app.include_router(router, prefix='/api')
    params = {'dataset_id': ID, 'expected_fingerprint': release.metadata['dataset_fingerprint'], 'year': 2020}
    with TestClient(app) as client:
        context = client.get('/api/analysis/earth/overview/context', params=params)
        assert context.status_code == 200, context.text
        assert context.json()['time']['time_zone'] == 'UTC'
        suite = client.get('/api/analysis/earth/overview/research-suite', params=params)
        assert suite.status_code == 200, suite.text[:1000]
        payload = suite.json()
        assert payload['day_count'] == 30
        assert payload['dates'][0] == '2020-01-01' and payload['dates'][-1] == '2020-01-30'
        assert payload['time_aggregation'] == THREE_HOUR_DAILY_AGGREGATION
        assert np.asarray(payload['seasonal']['TO3']['z']).shape == (60, 30)
        assert len(payload['latitude']) == 60
        spatial = client.get('/api/analysis/earth/overview/spatial-diagnostics', params={**params, 'variable': 'TO3'})
        assert spatial.status_code == 200, spatial.text[:1000]
        assert np.asarray(spatial.json()['anomaly']).shape == (60, 120)
        assert spatial.json()['source_grid_shape'] == [240, 480]
        assert len(spatial.content) < 250_000


def test_partial_utc_day_is_rejected_instead_of_averaged(research):
    service, _, release = research
    altered = replace(release, dates=release.dates[:-1])
    with pytest.raises(EarthOverviewError, match='eight consecutive') as error:
        service._threehour_variable_data(altered, 2020, 'TO3')
    assert error.value.code == 'incomplete_daily_aggregate'


def test_missing_native_sample_is_not_silently_a_partial_daily_mean(research, monkeypatch):
    service, _, release = research
    def missing(*args):
        values = np.ones((8, 240, 480), dtype='float32')
        values[0, 0, 0] = np.nan
        yield 0, values
    monkeypatch.setattr('services.earth_3hourly_overview.iter_threehour_chunks', missing)
    with pytest.raises(EarthOverviewError) as error:
        service._threehour_variable_data(release, 2020, 'TO3')
    assert (error.value.status_code, error.value.code) == (503, 'incomplete_daily_aggregate')


def test_research_cache_separates_dataset_and_version_even_with_same_fingerprint(research):
    service, _, release = research
    for change in ({'dataset_version': 'different'}, {'dataset_id': 'earth_merra2_daily_v2'}):
        metadata = {**release.metadata, **change}
        service._variable_cache[('sentinel', 'TO3')] = {}
        service._invalidate(replace(release, metadata=metadata))
        assert service._variable_cache == {}
