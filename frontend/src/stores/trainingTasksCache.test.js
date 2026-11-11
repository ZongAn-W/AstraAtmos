import test from 'node:test';
import assert from 'node:assert/strict';
import { createTrainingTasksCache, TRAINING_TASKS_CACHE_TTL_MS } from './trainingTasksCache.js';

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

test('fresh account summaries avoid another request and expire after sixty seconds', async () => {
  let time = 100;
  const cache = createTrainingTasksCache({ now: () => time });
  let calls = 0;
  const loader = async () => { calls++; return [{ id: calls }]; };
  const items = await cache.load('user:1', loader);
  time += TRAINING_TASKS_CACHE_TTL_MS - 1;
  assert.equal(await cache.load('user:1', loader), items);
  assert.equal(cache.peek('user:1'), items);
  assert.equal(calls, 1);
  time++;
  assert.equal(cache.peek('user:1'), null);
  assert.deepEqual(await cache.load('user:1', loader), [{ id: 2 }]);
});

test('different accounts never share cached or in-flight task lists', async () => {
  const cache = createTrainingTasksCache();
  const a = deferred();
  const b = deferred();
  const first = cache.load('user:1', () => a.promise);
  const second = cache.load('user:2', () => b.promise);
  a.resolve([{ owner: 1 }]); b.resolve([{ owner: 2 }]);
  await Promise.all([first, second]);
  assert.deepEqual(cache.peek('user:1'), [{ owner: 1 }]);
  assert.deepEqual(cache.peek('user:2'), [{ owner: 2 }]);
  cache.clear('user:1');
  assert.equal(cache.peek('user:1'), null);
  assert.deepEqual(cache.peek('user:2'), [{ owner: 2 }]);
});

test('simultaneous requests including forced refreshes share the active loader', async () => {
  const cache = createTrainingTasksCache();
  const pending = deferred();
  let calls = 0;
  const loader = () => { calls++; return pending.promise; };
  const first = cache.load('user:1', loader);
  const second = cache.load('user:1', loader, { force: true });
  const third = cache.load('user:1', loader);
  pending.resolve([]);
  assert.deepEqual(await Promise.all([first, second, third]), [[], [], []]);
  assert.equal(calls, 1);
});

test('canceling one subscriber leaves the other subscriber and shared request alive', async () => {
  const cache = createTrainingTasksCache();
  const pending = deferred();
  const subscriber = new AbortController();
  let sharedSignal;
  const first = cache.load('user:1', ({ signal }) => { sharedSignal = signal; return pending.promise; }, { signal: subscriber.signal });
  const second = cache.load('user:1', () => assert.fail('duplicate loader'));
  const rejected = assert.rejects(first, { name: 'AbortError' });
  await Promise.resolve();
  subscriber.abort();
  await rejected;
  assert.equal(sharedSignal.aborted, false);
  pending.resolve([{ id: 9 }]);
  assert.deepEqual(await second, [{ id: 9 }]);
  assert.deepEqual(cache.peek('user:1'), [{ id: 9 }]);
});

test('navigation can cancel all current subscribers and still warm the next screen', async () => {
  const cache = createTrainingTasksCache();
  const pending = deferred();
  const subscriber = new AbortController();
  const request = cache.load('user:1', () => pending.promise, { signal: subscriber.signal });
  const rejected = assert.rejects(request, { name: 'AbortError' });
  await Promise.resolve();
  subscriber.abort();
  await rejected;
  pending.resolve([{ id: 3 }]);
  assert.deepEqual(await cache.load('user:1', () => assert.fail('request should survive navigation')), [{ id: 3 }]);
});

test('logout and invalidation reject waiters and cannot be undone by a late response', async () => {
  for (const method of ['clear', 'invalidate']) {
    const cache = createTrainingTasksCache();
    const old = deferred();
    let sharedSignal;
    const request = cache.load('user:1', ({ signal }) => { sharedSignal = signal; return old.promise; });
    const rejected = assert.rejects(request, { name: 'AbortError' });
    await Promise.resolve();
    cache[method]('user:1');
    assert.equal(sharedSignal.aborted, true);
    const current = cache.load('user:1', async () => [{ id: 'new' }]);
    old.resolve([{ id: 'old' }]);
    await rejected;
    await current;
    assert.deepEqual(cache.peek('user:1'), [{ id: 'new' }]);
  }
});

test('an invalidated request finalizer cannot clear a newer pending operation', async () => {
  const cache = createTrainingTasksCache();
  const old = deferred();
  const next = deferred();
  const previous = cache.load('user:1', () => old.promise);
  const rejected = assert.rejects(previous, { name: 'AbortError' });
  await Promise.resolve();
  cache.invalidate('user:1');
  const current = cache.load('user:1', () => next.promise);
  await rejected;
  const shared = cache.load('user:1', () => assert.fail('new request lost its pending marker'));
  next.resolve([{ id: 'new' }]);
  assert.deepEqual(await current, [{ id: 'new' }]);
  assert.deepEqual(await shared, [{ id: 'new' }]);
  old.resolve([{ id: 'old' }]);
});

test('priming after a training refresh supersedes old results', async () => {
  const cache = createTrainingTasksCache();
  const old = deferred();
  const request = cache.load('user:1', () => old.promise);
  const rejected = assert.rejects(request, { name: 'AbortError' });
  await Promise.resolve();
  assert.equal(cache.prime('user:1', [{ id: 'current' }]), true);
  old.resolve([{ id: 'old' }]);
  await rejected;
  assert.deepEqual(await cache.load('user:1', () => assert.fail('already primed')), [{ id: 'current' }]);
});

test('errors and invalid payloads are retried instead of cached', async () => {
  const cache = createTrainingTasksCache();
  await assert.rejects(cache.load('user:1', async () => { throw new Error('offline'); }), /offline/);
  assert.equal(cache.peek('user:1'), null);
  await assert.rejects(cache.load('user:1', async () => ({ detail: 'bad response' })), TypeError);
  assert.equal(cache.peek('user:1'), null);
  assert.deepEqual(await cache.load('user:1', async () => []), []);
});

test('force bypasses completed cache while ordinary calls keep using fresh entries', async () => {
  const cache = createTrainingTasksCache();
  cache.prime('user:1', [{ id: 1 }]);
  assert.deepEqual(await cache.load('user:1', () => assert.fail('fresh')), [{ id: 1 }]);
  assert.deepEqual(await cache.load('user:1', async () => [{ id: 2 }], { force: true }), [{ id: 2 }]);
  assert.deepEqual(cache.peek('user:1'), [{ id: 2 }]);
});

test('unauthenticated scopes and already canceled calls never start a loader', async () => {
  const cache = createTrainingTasksCache();
  for (const scope of [null, '', 'anonymous', 'user:', 'user: 1', 1]) {
    assert.equal(cache.peek(scope), null);
    assert.equal(cache.prime(scope, []), false);
    await assert.rejects(cache.load(scope, () => assert.fail('private account required')), TypeError);
  }
  const controller = new AbortController();
  controller.abort();
  cache.prime('user:1', []);
  await assert.rejects(cache.load('user:1', () => assert.fail('already aborted'), { signal: controller.signal }), { name: 'AbortError' });
});
