import {
  EARTH_MODEL_ARCHITECTURE,
  EARTH_MODEL_SOURCE_OFFICIAL,
  EARTH_OPTIONAL_CHANNELS,
  getEarthSplitDefaults,
  isEarthTrainingDataset,
  resolveEarthTrainingRestore,
} from './earthTrainingConfig.js';
import {
  createDefaultArchitectureParamsByModel,
  sanitizeTrainingDataset,
} from './trainingParamSanitizers.js';
import { createDefaultCustomModelParams } from './uploadedModelParams.js';

const sceneKey = (dataset) => isEarthTrainingDataset(dataset) ? 'earth' : 'mars';
const paramsKey = (dataset, modelId) => `${sceneKey(dataset)}:${modelId || ''}`;
const SCENE_DEFAULT_FIELDS = new Set([
  'window', 'horizon', 'trainRatio', 'validationRatio', 'testRatio',
  'transferEnabled', 'transferFreezeMode', 'finetuneLearningRate',
]);

function clone(value) {
  if (Array.isArray(value)) return value.map(clone);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, clone(item)]));
  }
  return value;
}

function initialScene(dataset, defaults) {
  const earth = isEarthTrainingDataset(dataset);
  return {
    trainingDataset: dataset,
    modelSource: earth ? EARTH_MODEL_SOURCE_OFFICIAL : 'uploaded',
    selectedUploadedModelId: '',
    customModelParams: {},
    modelArchitecture: earth ? EARTH_MODEL_ARCHITECTURE : 'predrnnv2',
    useSphere: false,
    selectedChannels: earth ? [...EARTH_OPTIONAL_CHANNELS] : [],
    targetVariable: earth ? 'TO3' : 'O3',
    inputVariables: earth ? ['TO3', ...EARTH_OPTIONAL_CHANNELS] : ['O3'],
    hiddenDims: [64, 64, 64],
    stlstmLayers: 3,
    architectureParamsByModel: createDefaultArchitectureParamsByModel(),
    windowValue: defaults.window,
    horizon: defaults.horizon,
    ...(earth ? getEarthSplitDefaults(defaults) : {
      trainRatio: defaults.trainRatio,
      validationRatio: defaults.validationRatio,
      testRatio: defaults.testRatio,
    }),
    transferEnabled: earth ? false : defaults.transferEnabled,
    transferSourceType: 'task',
    transferSourceTaskId: '',
    selectedTrainingWeightId: '',
    transferFreezeMode: defaults.transferFreezeMode,
    finetuneLearningRate: defaults.finetuneLearningRate,
    transferStructureSnapshot: null,
    editedDefaultFields: [],
  };
}

/** Session-only scene transitions and uploaded-parameter ownership.
 *
 * The page applies one complete scene result. Snapshot storage, unchanged
 * selection, Earth constraints and schema arrival order stay behind this seam.
 * Epochs, batch size, optimizer settings and experiment name remain shared.
 */
export function createTrainingDraftSession() {
  let scenes = {};
  let parameterOwner = null;

  return {
    switchDataset(current, requestedDataset, defaults) {
      const dataset = sanitizeTrainingDataset(requestedDataset, { allowEarth: true });
      if (dataset === current.trainingDataset) return null;
      const previousScene = sceneKey(current.trainingDataset);
      const nextScene = sceneKey(dataset);
      scenes[previousScene] = clone(current);

      // Both Mars datasets use the same draft; changing the data does not
      // replace the model, custom split or transfer selection with defaults.
      let next = previousScene === nextScene
        ? clone(current)
        : clone(scenes[nextScene] || initialScene(dataset, defaults));
      if (nextScene === 'earth') {
        next = {
          ...next,
          ...resolveEarthTrainingRestore(next),
          transferSourceTaskId: '',
          selectedTrainingWeightId: '',
          transferStructureSnapshot: null,
        };
      }
      next.trainingDataset = dataset;
      next.editedDefaultFields = [
        ...(current.editedDefaultFields || []).filter((field) => !SCENE_DEFAULT_FIELDS.has(field)),
        ...(next.editedDefaultFields || []).filter((field) => SCENE_DEFAULT_FIELDS.has(field)),
      ];
      parameterOwner = paramsKey(dataset, next.selectedUploadedModelId);
      return next;
    },

    preserveCustomParams(dataset, modelId) {
      parameterOwner = paramsKey(dataset, modelId);
    },

    syncCustomParams(dataset, modelId, schema, current, locked = false) {
      const key = paramsKey(dataset, modelId);
      const defaults = createDefaultCustomModelParams(schema);
      const keep = locked || key === parameterOwner;
      parameterOwner = key;
      // Late or equivalent schemas may supply new defaults, but cannot replace
      // copied/restored/edited values. An explicit model change uses defaults.
      return keep ? { ...clone(defaults), ...clone(current) } : clone(defaults);
    },

    reset() {
      scenes = {};
      parameterOwner = null;
    },
  };
}
