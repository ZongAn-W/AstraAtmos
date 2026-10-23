import test from 'node:test';
import assert from 'node:assert/strict';

import { apiGetMe } from './api.js';
import {
  beginAuthenticatedPredictionSession,
  endAuthenticatedPredictionSession,
} from '../stores/authPredictionSession.js';
import {
  clearPredictCache,
  createUserPredictScope,
  getPredictCache,
  setPredictCache,
} from '../stores/predictCache.js';

const userScope = createUserPredictScope(101);

test.beforeEach(() => {
  endAuthenticatedPredictionSession();
  clearPredictCache(userScope);
});

test('API 401 clears the active user prediction cache before dispatching global logout', async () => {
  const operations = [];
  globalThis.localStorage = {
    getItem: () => 'expired-token',
    removeItem: (key) => operations.push(`remove:${key}`),
  };
  globalThis.window = {
    dispatchEvent: (event) => {
      operations.push(`event:${event.type}`);
      assert.equal(getPredictCache(userScope).results, null);
    },
  };
  globalThis.fetch = async () => ({
    ok: false,
    status: 401,
    statusText: 'Unauthorized',
    json: async () => ({}),
  });
  beginAuthenticatedPredictionSession(101);
  setPredictCache(userScope, {
    results: { owner: 'A' },
    compareTrainingPfiData: { owner: 'A' },
  });

  await assert.rejects(apiGetMe(), /401/);

  assert.deepEqual(operations, [
    'remove:astraatmos_token',
    'event:astraatmos:logout',
  ]);
  assert.equal(getPredictCache(userScope).results, null);
  assert.equal(getPredictCache(userScope).compareTrainingPfiData, null);
});
