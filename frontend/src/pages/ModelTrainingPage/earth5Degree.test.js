import test from 'node:test';
import assert from 'node:assert/strict';
import { getEarthTrainingProfile, buildEarthTrainingHyperparameters, getEarthUploadedSelectionBlocker } from './earthTrainingConfig.js';
import { sanitizeTrainingDataset } from './trainingParamSanitizers.js';
import { getEarthCadence, getEarthTrainingModelOptions } from '../PredictPage/earthPredictModel.js';

const datasetId = 'earth_merra2_3hourly_5deg_v1';
const schema = 'astraatmos_earth_3hourly_5deg_model_v1';
test('5 degree Earth preserves UTC cadence, windows and input variables', () => {
  assert.equal(sanitizeTrainingDataset(datasetId, { allowEarth: true }), datasetId);
  const profile = getEarthTrainingProfile(datasetId);
  assert.deepEqual(profile.gridShape, [36, 72]);
  assert.equal(profile.frequencyHours, 3);
  const payload = buildEarthTrainingHyperparameters({ datasetId, windowValue: 7, horizon: 1, selectedChannels: ['T2M'] });
  assert.equal(payload.training_dataset, datasetId);
  assert.equal(payload.window, 7);
  assert.equal(payload.horizon, 1);
  assert.deepEqual(payload.selected_channels, ['T2M']);
  assert.equal(getEarthCadence({ dataset_id: datasetId }).frequencyHours, 3);
  assert.equal(getEarthTrainingModelOptions([{ id: 1, status: 'completed', model_available: true, dataset_id: datasetId }]).length, 1);
});
test('uploaded compatibility is specific to the 5 degree dataset and schema', () => {
  const compatible = { package_id: 'model', dataset_id: datasetId, status: 'available', compatible: true,
    contract_schema: schema, datasets: { [datasetId]: { schema, window: [7], horizon: [1] } } };
  const selection = { datasetId, modelSource: 'uploaded', uploadedModelId: 'model', windowValue: 7, horizon: 1, compatibility: compatible };
  assert.equal(getEarthUploadedSelectionBlocker(selection), null);
  assert.ok(getEarthUploadedSelectionBlocker({ ...selection, compatibility: { ...compatible, dataset_id: 'earth_merra2_3hourly_v1' } }));
  assert.ok(getEarthUploadedSelectionBlocker({ ...selection, compatibility: { ...compatible, contract_schema: 'astraatmos_earth_3hourly_fullgrid_model_v2' } }));
});
