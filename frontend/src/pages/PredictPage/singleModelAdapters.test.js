import test from 'node:test';
import assert from 'node:assert/strict';
import { createEarthPredictionAdapter, createMarsPredictionAdapter, earthResponseMatchesRequest, marsResponseMatchesTask, normalizeEarthPrediction, utcInputValue, utcOriginValue } from './singleModelAdapters.js';

const field = value => ({ field: [[value, value + 1], [value + 2, value + 3]], minVal: value, maxVal: value + 3 });
const context = { task_id: 3, planet: 'earth', dataset_id: 'earth_merra2_3hourly_v1', dataset_version: 'v1', dataset_fingerprint: 'f',
  target_variable: 'TO3', input_variables: ['TO3', 'SWGDN'],
  target_unit: 'DU', window: 2, horizon: 2, grid: { latitude: [-41, 37], longitude: [-123, 89], latitude_range: [-90,90], longitude_range: [-180,180] },
  model: { model_source: 'official', model_architecture: 'dlinear' },
  origins: { timestamps: ['2021-01-01T01:30:00Z'], start: '2021-01-01T01:30:00Z', end: '2021-01-01T01:30:00Z' } };
const result = { ...context, forecast_origin: context.origins.start, target_timestamps: ['2021-01-01T04:30:00Z','2021-01-01T07:30:00Z'],
  reference: [field(300),field(301)], prediction: [field(300.125),field(301.125)], residual: [field(-1),field(-2)] };

test('Earth runs the complete task horizon and display steps carry lead hours and UTC', () => {
  const adapter = createEarthPredictionAdapter({ context });
  assert.deepEqual(adapter.request({taskId: 3, origin: context.origins.start, horizon: 1}), {trainingTaskId: 3, forecastOrigin: context.origins.start});
  assert.equal(adapter.horizon.editable, false);
  assert.equal(adapter.horizon.limit, 2);
  assert.equal(adapter.stepLabel(result, 1), '+6h · 2021-01-01T07:30:00Z UTC');
  assert.equal(adapter.unit, 'DU'); assert.equal(adapter.convertValue(300), 300);
  assert.equal(adapter.capabilities.diagnosticExport, false);
  assert.equal(adapter.capabilities.manualDiagnostics, true);
  assert.equal(utcOriginValue(utcInputValue(context.origins.start)), context.origins.start);
  assert.equal(adapter.origin.selectable('2021-01-01T02:30:00Z'), false);
});

test('Earth normalization preserves row/coordinate pairing, signed values and requested color palette', () => {
  const normalized = normalizeEarthPrediction(result, 'viridis');
  assert.deepEqual(normalized.ground_truth[0].field, result.reference[0].field);
  assert.deepEqual(normalized.prediction[0].lat, [-41,37]);
  assert.deepEqual(normalized.prediction[0].lon, [-123,89]);
  assert.equal(normalized.prediction[0].payload.colormap, 'viridis');
  assert.equal(normalized.residual[0].payload.colormap, 'rdbu');
  assert.deepEqual(normalized.residual[0].payload.color_range, {min:-2,max:2});
  assert.equal(createEarthPredictionAdapter().grid(normalized.prediction[0]).latitude[0], -41);
  const malformed = {...result, prediction: [{...field(300), field:[[300,301],[302]]},field(301)]};
  assert.equal(normalizeEarthPrediction(malformed).prediction[0], null);
});

test('old Earth responses are rejected after account, task, origin, artifact, grid or horizon changes', () => {
  const request = {taskId:3,origin:result.forecast_origin,context,scope:'user:1'};
  assert.equal(earthResponseMatchesRequest(result,request,'user:1'), true);
  assert.equal(earthResponseMatchesRequest(result,request,'user:2'), false);
  for (const changed of [{task_id:4},{forecast_origin:'old'},{dataset_fingerprint:'new'},{planet:'mars'},{horizon:1},{target_unit:'um-atm'},
    {grid:{...context.grid,latitude:[-40,40]}},{model:{...context.model,uploaded_model_content_hash:'new'}}]) {
    assert.equal(earthResponseMatchesRequest({...result,...changed},request,'user:1'),false,JSON.stringify(changed));
  }
});

test('Earth responses must keep the selected target and complete input order', () => {
  const request = { taskId: 3, origin: result.forecast_origin, context, scope: 'user:1' };
  assert.equal(earthResponseMatchesRequest({ ...result, target_variable: 'TO3', input_variables: ['TO3', 'SWGDN'] }, request, 'user:1'), true);
  assert.equal(earthResponseMatchesRequest({ ...result, target_variable: 'T2M', target_unit: 'K', input_variables: ['SWGDN'] }, request, 'user:1'), false);
  const partial = { ...result };
  delete partial.input_variables;
  assert.equal(earthResponseMatchesRequest(partial, request, 'user:1'), false);
});

test('Mars keeps its API body, horizon bounds, Ls labels and display-only DU conversion', () => {
  const adapter = createMarsPredictionAdapter({horizonLimit:7,ozoneUnit:'DU'});
  assert.deepEqual(adapter.request({taskId:4,origin:90,horizon:3,variables:['O3']}),
    {training_task_id:4,ls_start:90,horizon:3,selected_variables:['O3']});
  assert.equal(adapter.horizon.clamp(20),7);
  assert.equal(adapter.origin.selectable(356),false);
  assert.equal(adapter.convertValue(10),1);
  assert.match(adapter.stepLabel({ls_values:[94]},0), /Ls=94.000°/);
  assert.equal(adapter.capabilities.diagnosticExport,true);
});

test('Mars response metadata must match the selected task target and input order', () => {
  const task = { target_variable: 'T', input_variables: ['O3', 'U'] };
  assert.equal(marsResponseMatchesTask({ target_variable: 'T', target_unit: 'K', input_variables: ['O3', 'U'] }, { task }), true);
  assert.equal(marsResponseMatchesTask({ target_variable: 'O3', target_unit: 'um-atm', input_variables: ['O3', 'U'] }, { task }), false);
  assert.equal(marsResponseMatchesTask({ target_variable: 'O3', target_unit: 'um-atm', input_variables: ['O3'] }, { task: { target_variable: 'O3', input_variables: ['O3'] } }), true);
  assert.equal(marsResponseMatchesTask({ target_variable: 'T', target_unit: 'K' }, { task: { target_variable: 'T', target_unit: 'K', input_variables: ['O3'] } }), false);
  assert.equal(marsResponseMatchesTask({ target_variable: 'O3' }, { task: { target_variable: 'O3' } }), false);
  assert.equal(marsResponseMatchesTask({ ground_truth: [] }, { task: {} }), true);
  assert.equal(marsResponseMatchesTask({ target_unit: 'um-atm' }, { task: {} }), false);
});
