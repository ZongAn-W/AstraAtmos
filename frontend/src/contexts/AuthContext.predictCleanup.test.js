import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const currentDir = path.dirname(fileURLToPath(import.meta.url));
const source = fs.readFileSync(path.join(currentDir, 'AuthContext.jsx'), 'utf8');

test('AuthContext uses the shared prediction session for validated and new logins', () => {
  assert.match(source, /beginAuthenticatedPredictionSession/);
  assert.match(source, /apiGetMe\(\)[\s\S]*?beginAuthenticatedPredictionSession\(me\.id\)/);
  assert.match(source, /apiLogin\(email, password\)[\s\S]*?beginAuthenticatedPredictionSession\(data\.user\.id\)/);
});

test('manual logout and global logout events use the same prediction cleanup', () => {
  assert.match(
    source,
    /const handler = \(\) => \{[\s\S]*?endAuthenticatedPredictionSession\(\)[\s\S]*?setUser\(null\)/
  );
  assert.match(
    source,
    /const logout = useCallback\(\(\) => \{[\s\S]*?endAuthenticatedPredictionSession\(\)[\s\S]*?setUser\(null\)/
  );
});

test('missing and invalid startup tokens clear any outgoing prediction session before auth resolves', () => {
  assert.match(
    source,
    /if \(!stored\) \{[\s\S]*?endAuthenticatedPredictionSession\(\)[\s\S]*?setIsLoading\(false\)/
  );
  assert.match(
    source,
    /\.catch\(\(\) => \{[\s\S]*?endAuthenticatedPredictionSession\(\)[\s\S]*?localStorage\.removeItem\('astraatmos_token'\)/
  );
});
