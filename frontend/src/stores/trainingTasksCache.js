/** Private task summaries live only for this SPA session, isolated by account. */
export const TRAINING_TASKS_CACHE_TTL_MS = 60_000;

function validScope(scope) {
  return typeof scope === 'string' && /^user:\S+$/.test(scope);
}

function abortError() {
  const error = new Error('The request was aborted.');
  error.name = 'AbortError';
  return error;
}

// Subscribers own their cancellation; a navigation must not cancel a request
// still used by another screen, or prevent its result warming the cache.
function withAbort(promise, signal) {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(abortError());
  return new Promise((resolve, reject) => {
    const aborted = () => { cleanup(); reject(abortError()); };
    const cleanup = () => signal.removeEventListener('abort', aborted);
    signal.addEventListener('abort', aborted, { once: true });
    promise.then(
      (value) => { cleanup(); resolve(value); },
      (error) => { cleanup(); reject(error); },
    );
  });
}

export function createTrainingTasksCache({ now = Date.now, ttlMs = TRAINING_TASKS_CACHE_TTL_MS } = {}) {
  const entries = new Map();

  function peek(scope) {
    if (!validScope(scope)) return null;
    const entry = entries.get(scope);
    return entry?.items && now() < entry.expiresAt ? entry.items : null;
  }

  function clear(scope) {
    const entry = entries.get(scope);
    entries.delete(scope);
    // Delete before aborting: neither callbacks nor ignored aborts may restore
    // an invalidated account's private data.
    entry?.pending?.controller.abort();
  }

  function prime(scope, items) {
    if (!validScope(scope) || !Array.isArray(items)) return false;
    clear(scope);
    entries.set(scope, { items, expiresAt: now() + ttlMs, pending: null });
    return true;
  }

  function load(scope, loader, { signal, force = false } = {}) {
    if (!validScope(scope)) return Promise.reject(new TypeError('An authenticated user scope is required.'));
    if (signal?.aborted) return Promise.reject(abortError());
    let entry = entries.get(scope);
    if (entry?.pending) return withAbort(entry.pending.promise, signal);
    const cached = !force && peek(scope);
    if (cached) return withAbort(Promise.resolve(cached), signal);

    if (!entry) {
      entry = { items: null, expiresAt: 0, pending: null };
      entries.set(scope, entry);
    }
    const controller = new AbortController();
    const operation = { controller, promise: null };
    entry.pending = operation;
    const loaded = Promise.resolve().then(() => {
      if (controller.signal.aborted) throw abortError();
      return loader({ signal: controller.signal });
    });
    operation.promise = withAbort(loaded, controller.signal)
      .then((items) => {
        if (!Array.isArray(items)) throw new TypeError('Training tasks must be an array.');
        if (entries.get(scope) === entry && entry.pending === operation) {
          entry.items = items;
          entry.expiresAt = now() + ttlMs;
        }
        return items;
      })
      .catch((error) => {
        if (entries.get(scope) === entry && entry.pending === operation) entries.delete(scope);
        throw error;
      })
      .finally(() => {
        if (entries.get(scope) === entry && entry.pending === operation) entry.pending = null;
      });
    return withAbort(operation.promise, signal);
  }

  return { peek, load, invalidate: clear, clear, prime };
}

export const trainingTasksCache = createTrainingTasksCache();
