import test from 'node:test';
import assert from 'node:assert/strict';

import {
  apiGetMe,
  fetchTasks,
  renameTrainingModel,
  stopTrainingTask,
  deleteTrainingTask,
  createTrainingTag,
} from './api.js';
import {
  beginAuthenticatedPredictionSession,
  endAuthenticatedPredictionSession,
  getActiveAuthenticatedPredictScope,
} from '../stores/authPredictionSession.js';
import { trainingTasksCache } from '../stores/trainingTasksCache.js';

const originalFetch = globalThis.fetch;
const originalStorage = globalThis.localStorage;
const originalWindow = globalThis.window;
let requests;

function response(items, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => items };
}

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

test.beforeEach(() => {
  endAuthenticatedPredictionSession();
  trainingTasksCache.clear('user:101');
  trainingTasksCache.clear('user:202');
  requests = [];
  // No credentials or persistent browser storage are needed for these mocks.
  globalThis.localStorage = { getItem: () => null, removeItem() {} };
  globalThis.window = { dispatchEvent() {} };
  beginAuthenticatedPredictionSession(101);
});

test.afterEach(() => {
  endAuthenticatedPredictionSession();
  trainingTasksCache.clear('user:101');
  trainingTasksCache.clear('user:202');
  globalThis.fetch = originalFetch;
  if (originalStorage === undefined) delete globalThis.localStorage;
  else globalThis.localStorage = originalStorage;
  if (originalWindow === undefined) delete globalThis.window;
  else globalThis.window = originalWindow;
});

test('reentering prediction reuses task summaries without another request', async () => {
  globalThis.fetch = async (url, options) => {
    requests.push({ url, options });
    return response([{ id: 7, status: 'completed', model_available: true, dataset_snapshot: null }]);
  };

  const firstVisit = await fetchTasks();
  const nextVisit = await fetchTasks();

  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, '/api/training/tasks?summary=true');
  assert.equal(nextVisit, firstVisit);
});

test('an empty successful catalog is cached and force refresh replaces it', async () => {
  globalThis.fetch = async (url, options) => {
    requests.push({ url, options });
    return response(requests.length === 1 ? [] : [{ id: 8 }]);
  };

  assert.deepEqual(await fetchTasks(), []);
  assert.deepEqual(await fetchTasks(), []);
  assert.equal(requests.length, 1);
  assert.deepEqual(await fetchTasks({ force: true }), [{ id: 8 }]);
  assert.deepEqual(await fetchTasks(), [{ id: 8 }]);
  assert.equal(requests.length, 2);
  assert.equal(requests[1].options.force, undefined);
});

test('navigation cancellation leaves a shared catalog request usable by other callers', async () => {
  const started = deferred();
  const received = deferred();
  globalThis.fetch = async (url, options) => {
    requests.push({ url, options });
    started.resolve();
    return received.promise;
  };
  const navigation = new AbortController();
  const leavingPage = fetchTasks({ signal: navigation.signal });
  const remainingPage = fetchTasks();
  const cancelled = assert.rejects(leavingPage, { name: 'AbortError' });
  await started.promise;
  navigation.abort();
  assert.equal(requests[0].options.signal.aborted, false);
  received.resolve(response([{ id: 9 }]));

  await cancelled;
  assert.deepEqual(await remainingPage, [{ id: 9 }]);
  assert.deepEqual(await fetchTasks(), [{ id: 9 }]);
  assert.equal(requests.length, 1);
});

for (const [name, mutate] of [
  ['rename', () => renameTrainingModel(7, 'Renamed experiment')],
  ['stop', () => stopTrainingTask(7)],
  ['delete', () => deleteTrainingTask(7)],
  ['tag edit', () => createTrainingTag('Baseline')],
]) {
  test(`a successful ${name} invalidates the cached catalog`, async () => {
    let catalogVersion = 1;
    globalThis.fetch = async (url, options) => {
      requests.push({ url, options });
      if (url === '/api/training/tasks?summary=true') return response([{ id: 7, version: catalogVersion }]);
      catalogVersion = 2;
      return response({ id: 7 });
    };

    assert.equal((await fetchTasks())[0].version, 1);
    await mutate();
    assert.equal((await fetchTasks())[0].version, 2);
    assert.equal(requests.filter(({ url }) => url === '/api/training/tasks?summary=true').length, 2);
  });
}

test('a rejected mutation leaves the previously confirmed catalog available', async () => {
  globalThis.fetch = async (url, options) => {
    requests.push({ url, options });
    return url === '/api/training/tasks?summary=true'
      ? response([{ id: 7 }])
      : response({ detail: 'This experiment cannot be renamed' }, 409);
  };

  const confirmed = await fetchTasks();
  await assert.rejects(renameTrainingModel(7, 'Rejected'), /cannot be renamed/);
  assert.equal(await fetchTasks(), confirmed);
  assert.equal(requests.length, 2);
});

test('logout and a subsequent login do not reuse the previous session catalog', async () => {
  globalThis.fetch = async (url) => {
    requests.push(url);
    return response([{ id: requests.length }]);
  };

  assert.deepEqual(await fetchTasks(), [{ id: 1 }]);
  endAuthenticatedPredictionSession();
  beginAuthenticatedPredictionSession(101);
  assert.deepEqual(await fetchTasks(), [{ id: 2 }]);
  assert.equal(requests.length, 2);
});

test('switching accounts never shares the outgoing user catalog', async () => {
  globalThis.fetch = async (url) => {
    requests.push(url);
    return response([{ account: getActiveAuthenticatedPredictScope() }]);
  };

  assert.deepEqual(await fetchTasks(), [{ account: 'user:101' }]);
  beginAuthenticatedPredictionSession(202);
  assert.deepEqual(await fetchTasks(), [{ account: 'user:202' }]);
  beginAuthenticatedPredictionSession(101);
  assert.deepEqual(await fetchTasks(), [{ account: 'user:101' }]);
  assert.equal(requests.length, 3);
});

test('a late outgoing account response cannot populate the incoming user cache', async () => {
  const started = deferred();
  const oldResponse = deferred();
  globalThis.fetch = async (url, options) => {
    requests.push({ url, options });
    if (requests.length === 1) {
      started.resolve();
      return oldResponse.promise; // Deliberately ignore network abort to exercise the guard.
    }
    return response([{ account: 'B' }]);
  };
  const outgoingRequest = fetchTasks();
  const cancelled = assert.rejects(outgoingRequest, { name: 'AbortError' });
  await started.promise;
  beginAuthenticatedPredictionSession(202);
  assert.deepEqual(await fetchTasks(), [{ account: 'B' }]);
  oldResponse.resolve(response([{ account: 'A' }]));
  await cancelled;
  assert.deepEqual(await fetchTasks(), [{ account: 'B' }]);
  assert.equal(requests.length, 2);
});

test('a 401 on another authenticated API clears task summaries before logout', async () => {
  globalThis.fetch = async (url) => url === '/api/auth/me'
    ? response({}, 401)
    : response([{ id: 7 }]);
  await fetchTasks();
  assert.ok(trainingTasksCache.peek('user:101'));

  await assert.rejects(apiGetMe(), /401/);

  assert.equal(getActiveAuthenticatedPredictScope(), null);
  assert.equal(trainingTasksCache.peek('user:101'), null);
});

test('a late 401 from the outgoing account cannot log out the incoming session', async () => {
  const started = deferred();
  const oldResponse = deferred();
  globalThis.fetch = async (url) => {
    if (url === '/api/auth/me') {
      started.resolve();
      return oldResponse.promise;
    }
    requests.push(url);
    return response([{ account: 'B' }]);
  };
  const oldAccountRequest = apiGetMe();
  const rejected = assert.rejects(oldAccountRequest, /401/);
  await started.promise;
  beginAuthenticatedPredictionSession(202);
  const newCatalog = await fetchTasks();
  oldResponse.resolve(response({}, 401));
  await rejected;

  assert.equal(getActiveAuthenticatedPredictScope(), 'user:202');
  assert.equal(await fetchTasks(), newCatalog);
  assert.equal(requests.length, 1);
});

for (const [name, firstResponse] of [
  ['HTTP failure', () => response({}, 503)],
  ['malformed catalog', () => response({ tasks: [] })],
]) {
  test(`${name} is not cached and can be retried`, async () => {
    globalThis.fetch = async (url) => {
      requests.push(url);
      return requests.length === 1 ? firstResponse() : response([{ id: 10 }]);
    };

    await assert.rejects(fetchTasks());
    assert.deepEqual(await fetchTasks(), [{ id: 10 }]);
    assert.deepEqual(await fetchTasks(), [{ id: 10 }]);
    assert.equal(requests.length, 2);
  });
}
