import { clearPredictCache, createUserPredictScope } from './predictCache.js';

let activeAuthenticatedScope = null;

// A response can pass through more than one error-handling layer (for example
// a shared wrapper and a feature-specific parser). Keep the cleanup side effect
// idempotent for that response while allowing later 401 responses to be handled.
const handledUnauthorizedResponses = new WeakSet();

export function getActiveAuthenticatedPredictScope() {
  return activeAuthenticatedScope;
}

export function beginAuthenticatedPredictionSession(userId) {
  const nextScope = createUserPredictScope(userId);
  if (!nextScope) {
    endAuthenticatedPredictionSession();
    return null;
  }

  if (activeAuthenticatedScope && activeAuthenticatedScope !== nextScope) {
    clearPredictCache(activeAuthenticatedScope);
  }
  activeAuthenticatedScope = nextScope;
  return nextScope;
}

export function endAuthenticatedPredictionSession() {
  if (activeAuthenticatedScope) clearPredictCache(activeAuthenticatedScope);
  activeAuthenticatedScope = null;
}

/**
 * Clear the private prediction session and notify AuthContext when an
 * authenticated request is rejected. Returns true only for HTTP 401.
 */
export function handleUnauthorizedResponse(response) {
  if (!response || response.status !== 401) return false;

  if (typeof response === 'object' && handledUnauthorizedResponses.has(response)) return true;
  if (typeof response === 'object') handledUnauthorizedResponses.add(response);

  endAuthenticatedPredictionSession();
  if (typeof localStorage !== 'undefined') localStorage.removeItem('astraatmos_token');
  if (typeof window !== 'undefined' && typeof window.dispatchEvent === 'function') {
    const event = typeof Event === 'function'
      ? new Event('astraatmos:logout')
      : { type: 'astraatmos:logout' };
    window.dispatchEvent(event);
  }
  return true;
}
