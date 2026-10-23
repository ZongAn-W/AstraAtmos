import test from 'node:test';
import assert from 'node:assert/strict';

import {
  compareEarthModels,
  fetchEarthDiagnosticContext,
  fetchEarthPredictContext,
  EarthPredictApiError,
  runEarthDiagnostics,
  runEarthPrediction,
} from './earthPredict.js';
import {
  endAuthenticatedPredictionSession,
  handleUnauthorizedResponse,
} from '../stores/authPredictionSession.js';

const earthRequests = [
  {
    name: 'fetchEarthPredictContext',
    call: () => fetchEarthPredictContext(31),
  },
  {
    name: 'runEarthPrediction',
    call: () => runEarthPrediction({ trainingTaskId: 31, forecastOrigin: '2020-01-01T00:00:00Z' }),
  },
  {
    name: 'compareEarthModels',
    call: () => compareEarthModels([31, 32]),
  },
  {
    name: 'fetchEarthDiagnosticContext',
    call: () => fetchEarthDiagnosticContext(31),
  },
  {
    name: 'runEarthDiagnostics',
    call: () => runEarthDiagnostics({ training_task_id: 31 }),
  },
];

function installEnvironment(fetchImpl) {
  const operations = [];
  globalThis.localStorage = {
    token: 'expired-token',
    getItem(key) {
      return key === 'astraatmos_token' ? this.token : null;
    },
    removeItem(key) {
      if (key === 'astraatmos_token') {
        this.token = null;
        operations.push(`remove:${key}`);
      }
    },
  };
  globalThis.window = {
    dispatchEvent(event) {
      operations.push(`event:${event.type}`);
    },
  };
  globalThis.fetch = fetchImpl;
  return { operations, storage: globalThis.localStorage };
}

function unauthorizedResponse() {
  return {
    ok: false,
    status: 401,
    statusText: 'Unauthorized',
    json: async () => ({ detail: { code: 'token_expired', message: 'Session expired' } }),
  };
}

test.beforeEach(() => {
  endAuthenticatedPredictionSession();
});

for (const { name, call } of earthRequests) {
  test(`${name} handles 401 with one global cleanup event`, async () => {
    const { operations, storage } = installEnvironment(async () => unauthorizedResponse());

    await assert.rejects(call(), (error) => {
      assert.ok(error instanceof EarthPredictApiError);
      assert.equal(error.status, 401);
      assert.equal(error.code, 'token_expired');
      assert.equal(error.message, 'Session expired');
      return true;
    });

    assert.equal(storage.getItem('astraatmos_token'), null);
    assert.deepEqual(operations, [
      'remove:astraatmos_token',
      'event:astraatmos:logout',
    ]);
  });
}

test('Earth 401 cleanup prevents a later request from carrying the old Authorization header', async () => {
  const requests = [];
  const { operations, storage } = installEnvironment(async (url, options) => {
    requests.push({ url, options });
    return requests.length === 1
      ? unauthorizedResponse()
      : { ok: true, status: 200, json: async () => ({ ok: true }) };
  });

  await assert.rejects(fetchEarthPredictContext(31), (error) => error.status === 401);
  assert.equal(storage.getItem('astraatmos_token'), null);
  await fetchEarthPredictContext(31);

  assert.equal(requests[0].options.headers.Authorization, 'Bearer expired-token');
  assert.equal(requests[1].options.headers.Authorization, undefined);
  assert.deepEqual(operations, [
    'remove:astraatmos_token',
    'event:astraatmos:logout',
  ]);
});

for (const status of [409, 422, 500]) {
  test(`Earth structured non-401 ${status} errors preserve code, status and message`, async () => {
    installEnvironment(async () => ({
      ok: false,
      status,
      statusText: status === 409 ? 'Conflict' : status === 422 ? 'Unprocessable Entity' : 'Server Error',
      json: async () => ({ detail: { code: 'invalid_origin', message: 'Forecast origin is outside the dataset' } }),
    }));

    await assert.rejects(
      runEarthPrediction({ trainingTaskId: 31, forecastOrigin: 'invalid' }),
      (error) => {
        assert.ok(error instanceof EarthPredictApiError);
        assert.equal(error.code, 'invalid_origin');
        assert.equal(error.status, status);
        assert.equal(error.message, 'Forecast origin is outside the dataset');
        return true;
      },
    );
  });
}

test('the same 401 response is cleaned up and broadcast only once', () => {
  const { operations } = installEnvironment(async () => unauthorizedResponse());
  const response = unauthorizedResponse();

  assert.equal(handleUnauthorizedResponse(response), true);
  assert.equal(handleUnauthorizedResponse(response), true);
  assert.deepEqual(operations, [
    'remove:astraatmos_token',
    'event:astraatmos:logout',
  ]);
});

test('Earth AbortError propagates without clearing authentication or dispatching logout', async () => {
  const { operations, storage } = installEnvironment(async () => {
    const error = new Error('Cancelled');
    error.name = 'AbortError';
    throw error;
  });

  await assert.rejects(fetchEarthDiagnosticContext(31), (error) => error.name === 'AbortError');
  assert.equal(storage.getItem('astraatmos_token'), 'expired-token');
  assert.deepEqual(operations, []);
});
