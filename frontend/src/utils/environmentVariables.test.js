import test from 'node:test';
import assert from 'node:assert/strict';
import { environmentDisplay, resolveEnvironmentTask, sameEnvironmentTarget } from './environmentVariables.js';
import { buildEarthTrainingHyperparameters, getEarthUploadedSelectionBlocker } from '../pages/ModelTrainingPage/earthTrainingConfig.js';
import { createTrainingDraftSession } from '../pages/ModelTrainingPage/trainingDraftSession.js';
import { readExperimentConfig } from '../pages/ModelTrainingPage/experimentCenterModel.js';
import { createMarsPredictionPresentation, createEarthPredictionPresentation, predictionMetricCards } from '../pages/PredictPage/predictionPresentation.js';
import { createEarthPredictionAdapter, createMarsPredictionAdapter } from '../pages/PredictPage/singleModelAdapters.js';
import { earthDiagnosticMatchesIdentity, earthDiagnosticScatter } from '../pages/PredictPage/earthDiagnosticsModel.js';

test('historical tasks retain ozone-first inputs; explicit inputs can exclude the target', () => {
  assert.deepEqual(resolveEnvironmentTask({ selected_channels: ['T'] }).inputs, ['O3', 'T']);
  const task = { dataset_id: 'earth_merra2_3hourly_v1', hyperparameters: JSON.stringify({ target_variable: 'T2M', input_variables: ['TO3', 'SWGDN'] }) };
  assert.deepEqual(resolveEnvironmentTask(task).inputs, ['TO3', 'SWGDN']);
  assert.equal(resolveEnvironmentTask(task).unit, 'K');
  assert.equal(resolveEnvironmentTask(task).target, 'T2M');
});

test('temperature display keeps Kelvin even when the ozone preference is DU', () => {
  const mars = environmentDisplay({ target_variable: 'T', input_variables: ['O3', 'S'] }, 'mars', 'DU');
  assert.equal(mars.unit, 'K');
  assert.equal(mars.convertValue(210), 210);
  assert.equal(environmentDisplay({}, 'mars', 'DU').convertValue(210), 21);
  const adapter = createMarsPredictionAdapter({ task: { target_variable: 'T' }, ozoneUnit: 'DU' });
  assert.equal(adapter.unit, 'K');
  assert.equal(adapter.convertValue(210), 210);
});

test('Earth submits the chosen target and exact input set without restoring target history', () => {
  const payload = buildEarthTrainingHyperparameters({ targetVariable: 'T2M', inputVariables: ['TO3', 'SWGDN'] });
  assert.equal(payload.target_variable, 'T2M');
  assert.deepEqual(payload.input_variables, ['TO3', 'SWGDN']);
});

test('copy preserves explicit variables and reconstructs legacy Earth auxiliary inputs', () => {
  const base = { id: 1, dataset_id: 'earth_merra2_3hourly_v1', model_script: 'demo3.py' };
  const task = { ...base, hyperparameters: { training_dataset: base.dataset_id, target_variable: 'T2M', input_variables: ['TO3', 'SWGDN'] } };
  const copy = readExperimentConfig(task, { channelOrder: ['U', 'V', 'D', 'S', 'T'] });
  assert.equal(copy.targetVariable, 'T2M');
  assert.deepEqual(copy.inputVariables, ['TO3', 'SWGDN']);
  assert.deepEqual(readExperimentConfig({ ...base, hyperparameters: { training_dataset: base.dataset_id, selected_channels: ['T2M'] } }, { channelOrder: ['U', 'T'] }).inputVariables, ['TO3', 'T2M']);
});

test('scene round trips preserve independent Mars and Earth forecast definitions', () => {
  const session = createTrainingDraftSession();
  const defaults = { window: 8, horizon: 4, trainRatio: .7, validationRatio: .2, testRatio: .1 };
  const mars = { trainingDataset: 'mcd_overview', targetVariable: 'T', inputVariables: ['O3', 'S'], selectedChannels: ['S'] };
  const earth = session.switchDataset(mars, 'earth_merra2_3hourly_v1', defaults);
  earth.targetVariable = 'SWGDN'; earth.inputVariables = ['T2M'];
  const restoredMars = session.switchDataset(earth, 'mcd_overview', defaults);
  assert.equal(restoredMars.targetVariable, 'T');
  assert.deepEqual(restoredMars.inputVariables, ['O3', 'S']);
  const restoredEarth = session.switchDataset(restoredMars, 'earth_merra2_3hourly_v1', defaults);
  assert.equal(restoredEarth.targetVariable, 'SWGDN');
  assert.deepEqual(restoredEarth.inputVariables, ['T2M']);
});

test('Kelvin metrics, scatter and Earth adapters never apply ozone conversion', () => {
  const mars = createMarsPredictionPresentation({ task: { target_variable: 'T' }, ozoneUnit: 'DU' });
  assert.equal(predictionMetricCards({ rmse: 5 }, mars.metrics)[0].value, 5);
  const earth = createEarthPredictionPresentation({ context: { target_variable: 'T2M', target_unit: 'K' } });
  assert.equal(earth.metrics.items.find(item => item.key === 'mse').unit, 'K²');
  assert.equal(earth.metrics.items.find(item => item.key === 'mae').unit, 'K');
  assert.equal(predictionMetricCards({ mse: 4, metric_units: { mse: '(m s-1)^2' } }, earth.metrics)[0].unit, '(m s-1)^2');
  assert.equal(createEarthPredictionAdapter({ context: { target_variable: 'T2M' } }).unit, 'K');
  assert.deepEqual(earthDiagnosticScatter({ reference: [280], prediction: [281], unit: 'K' }, 'K').reference, [280]);
  assert.equal(earthDiagnosticMatchesIdentity({ target_variable: 'T2M', target_unit: 'K' }, { target_variable: 'TO3', target_unit: 'DU' }), false);
});

test('uploaded forecasts require explicit supported-target and input evidence', () => {
  const compatibility = { compatible: true, status: 'available', dataset_id: 'earth_merra2_3hourly_v1', contract_schema: 'astraatmos_earth_3hourly_uploaded_model_v1' };
  const base = { modelSource: 'uploaded', uploadedModelId: 'm', compatibility };
  assert.equal(getEarthUploadedSelectionBlocker({ ...base, targetVariable: 'T2M', inputVariables: ['TO3'] }), 'forecast_target_not_supported');
  assert.equal(getEarthUploadedSelectionBlocker({ ...base, inputVariables: ['T2M'] }), 'forecast_inputs_require_to3');
  const extended = { ...compatibility, supported_targets: ['T2M'], supports_independent_inputs: true, input_variable_sets: [['TO3', 'SWGDN']] };
  assert.equal(getEarthUploadedSelectionBlocker({ ...base, compatibility: extended, targetVariable: 'T2M', inputVariables: ['TO3', 'SWGDN'] }), null);
  assert.equal(getEarthUploadedSelectionBlocker({ ...base, compatibility: extended, targetVariable: 'T2M', inputVariables: ['T2M'] }), 'forecast_input_combination_not_supported');
});

test('comparisons cannot mix different physical targets', () => {
  assert.equal(sameEnvironmentTarget([{ target_variable: 'T' }, { target_variable: 'O3' }]), false);
  assert.equal(sameEnvironmentTarget([{ target_variable: 'T' }, { target_variable: 'T' }]), true);
});
