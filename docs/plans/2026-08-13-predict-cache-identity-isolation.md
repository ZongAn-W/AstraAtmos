# Predict Cache Identity Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate all runtime prediction state by stable user identity and synchronously remove outgoing-user state on every authentication termination path.

**Architecture:** Replace the module singleton with explicit in-memory scopes and a separate presentation-preference object. A small authentication-session coordinator owns the active authenticated scope so manual logout, API 401, startup validation failure, and user replacement share one cleanup operation without importing React authentication code into the cache. Prediction consumers derive scope from resolved authentication state, reset before paint on scope changes, validate request context and task access before restoration, and reject stale async completions.

**Tech Stack:** React 19, Vite 6, ES modules, Node built-in test runner (`node --test`).

---

### Task 1: Explicit identity-scoped prediction cache

**Files:**
- Modify: `frontend/src/stores/predictCache.js`
- Modify: `frontend/src/stores/predictCache.test.js`

- [ ] **Step 1: Write failing scope-isolation tests**

Extend the cache tests to use the wished-for API:

```js
const userA = createUserPredictScope(101);
const userB = createUserPredictScope(202);

setPredictCache(userA, { results: { owner: 'A' }, trainingTaskId: 7 });
assert.equal(getPredictCache(userB).results, null);
assert.equal(getPredictCache(ANONYMOUS_PREDICT_SCOPE).results, null);

setPredictCache(ANONYMOUS_PREDICT_SCOPE, { results: { owner: 'anonymous' } });
assert.equal(getPredictCache(userA).results.owner, 'A');
```

Also assert that `null` scope cannot read/write, clearing user A leaves user B untouched, exact restore checks both scope and context, and `viewMode` remains after sensitive cache clearing.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
node --test frontend/src/stores/predictCache.test.js
```

Expected: FAIL because scope constants/helpers and scoped method signatures do not exist.

- [ ] **Step 3: Implement the scoped cache**

In `predictCache.js`:

```js
export const ANONYMOUS_PREDICT_SCOPE = 'anonymous';

export function createUserPredictScope(userId) {
  const normalized = String(userId ?? '').trim();
  return normalized ? `user:${normalized}` : null;
}

export function resolvePredictCacheScope({ user, isLoading }) {
  if (isLoading) return null;
  return user?.id != null ? createUserPredictScope(user.id) : ANONYMOUS_PREDICT_SCOPE;
}
```

Store sensitive snapshots in `Map<string, object>`, keep `viewMode` in a separate UI-preferences object, return fresh defaults for missing/invalid scopes, require `setPredictCache(scope, updates)`, and require `clearPredictCache(scope)`. Include all existing result, metric, error, PFI, comparison, model/task, parameter, key, workflow graph/config, error, and loading-related cache fields in the empty scoped template. Return snapshots with `scope`, and change the context helper to require `expectedScope` plus exact `contextKey`.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Task 1 test command. Expected: all cache tests pass.

### Task 2: Shared authentication-session cleanup

**Files:**
- Create: `frontend/src/stores/authPredictionSession.js`
- Create: `frontend/src/stores/authPredictionSession.test.js`
- Modify: `frontend/src/contexts/AuthContext.jsx`
- Create: `frontend/src/contexts/AuthContext.predictCleanup.test.js`
- Modify: `frontend/src/services/api.js`
- Create: `frontend/src/services/api.authCleanup.test.js`

- [ ] **Step 1: Write failing behavioral and integration tests**

Test the session coordinator API before it exists:

```js
beginAuthenticatedPredictionSession(101);
setPredictCache(createUserPredictScope(101), { results: { owner: 'A' } });
endAuthenticatedPredictionSession();
assert.equal(getPredictCache(createUserPredictScope(101)).results, null);
```

Test changing from user A to user B clears A. In `api.authCleanup.test.js`, install fake `localStorage`, `window.dispatchEvent`, and a `fetch` returning 401; seed the active user cache, call a protected API, then assert token removal, logout event dispatch, and cache removal. Add a source-structure test asserting active logout, global logout handler, and startup `apiGetMe().catch` all call the shared session cleanup rather than only clearing local React state.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
node --test frontend/src/stores/authPredictionSession.test.js frontend/src/services/api.authCleanup.test.js frontend/src/contexts/AuthContext.predictCleanup.test.js
```

Expected: FAIL because the session coordinator and shared calls do not exist.

- [ ] **Step 3: Implement the independent session coordinator**

Create a module that imports only cache helpers:

```js
let activeAuthenticatedScope = null;

export function beginAuthenticatedPredictionSession(userId) {
  const nextScope = createUserPredictScope(userId);
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
```

`AuthContext` calls `begin...` before exposing a successfully validated/logged-in user. Manual logout, logout events, missing-token initialization, and initialization failure call `end...` before clearing state or completing loading. `api.js` calls `end...` synchronously before dispatching `astraatmos:logout` on 401. No token enters either cache/session module.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Task 2 command plus `frontend/src/services/api.predictAbort.test.js`. Expected: all pass.

### Task 3: Scope-aware prediction consumers and immediate page reset

**Files:**
- Modify: `frontend/src/pages/PredictPage.jsx`
- Create: `frontend/src/pages/PredictPage/predictIdentityIsolationStructure.test.js`
- Modify: `frontend/src/pages/AIPage.jsx`
- Modify: `frontend/src/pages/PredictPage/WorkflowCanvas/WorkflowCanvas.jsx`

- [ ] **Step 1: Write failing page and consumer structure tests**

Assert that `PredictPage` derives scope from `{ user, isLoading }`, starts with empty sensitive state, uses `useLayoutEffect` for scope transitions, clears each listed state setter, passes scope to every cache read/write, and includes the request-start scope in request-current checks. Assert `AIPage` returns an empty snapshot during auth loading and refreshes using its resolved scope. Assert `WorkflowCanvas` supplies its resolved scope to both cache writes.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
node --test frontend/src/pages/PredictPage/predictIdentityIsolationStructure.test.js
```

Expected: FAIL because consumers still use the unscoped singleton.

- [ ] **Step 3: Implement resolved-scope lifecycle in `PredictPage`**

Initialize sensitive React state from empty constants, never from cache during render. Derive `predictScope`; keep it in a ref. On scope change in `useLayoutEffect`, invalidate all requests, clear loading/error/result/analysis/model/task/comparison/key/fullscreen state, clear an outgoing `user:*` cache, set the new scope, and restore only the new snapshot. While scope is null, restore nothing and cache writes are rejected.

Wrap every cache write with the active scope. Add the starting scope to every request token and require it to match the active scope in both `isRequestCurrent` and `isRequestLatest`. This makes post-logout completions unable to update state or cache.

- [ ] **Step 4: Update the other cache consumers**

`AIPage` uses `useAuth` and `resolvePredictCacheScope`; refresh calls `getPredictCache(scope)` and dependencies include the scope. `WorkflowCanvas` resolves its scope and calls `setPredictCache(scope, updates)`; unresolved authentication performs no write.

- [ ] **Step 5: Run tests and verify GREEN**

Run the Task 3 test plus the existing prediction request coordinator, request consistency, cache key, horizon, and workflow tests. Expected: all pass.

### Task 4: Training-task authorization validation and complete regression verification

**Files:**
- Create: `frontend/src/pages/PredictPage/predictCacheTaskValidation.js`
- Create: `frontend/src/pages/PredictPage/predictCacheTaskValidation.test.js`
- Modify: `frontend/src/pages/PredictPage.jsx`
- Modify: `frontend/src/pages/PredictPage/predictIdentityIsolationStructure.test.js`

- [ ] **Step 1: Write failing task-validation tests**

Define and test a pure validator that receives a scoped snapshot and current completed task IDs. For an inaccessible selected task, assert it returns `trainingTaskId: null` and nulls the single result, metrics, error distribution, PFI, performance, result context, and analysis keys. For comparison IDs containing an inaccessible task, assert selections are filtered and all multi-model metrics/error/PFI data and keys are null. Valid task IDs preserve exact-context data.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
node --test frontend/src/pages/PredictPage/predictCacheTaskValidation.test.js
```

Expected: FAIL because the validator does not exist.

- [ ] **Step 3: Implement and integrate task validation**

Create `validatePredictCacheTrainingTasks(snapshot, accessibleTaskIds)`. In `PredictPage`, do not expose trained-task result or comparison bundles before the current authenticated user's task list finishes loading. Once loaded, validate the scoped snapshot, write the sanitized snapshot back to the same scope, and restore only validated selections/results. Anonymous snapshots never restore authenticated training-task selections.

- [ ] **Step 4: Verify all affected tests and build**

Run:

```powershell
node --test frontend/src/stores/*.test.js frontend/src/services/*.test.js frontend/src/contexts/*.test.js frontend/src/pages/PredictPage/*.test.js frontend/src/pages/PredictPage/CompareTrainingModels/*.test.js frontend/src/pages/PredictPage/WorkflowCanvas/*.test.js
npm run build --prefix frontend
git diff --check
```

Expected: zero test failures, Vite build exit code 0, and no whitespace errors.

- [ ] **Step 5: Review the final diff against the ten design test cases**

Confirm all cache calls are explicit-scope, no token is cached or used as a key, no prediction result is persisted, the outgoing scope is cleared on all authentication endings, and existing user request-cancellation edits remain present.
