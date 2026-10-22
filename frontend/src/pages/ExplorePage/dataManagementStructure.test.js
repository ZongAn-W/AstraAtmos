import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const explorePage = readFileSync(resolve(__dirname, '../ExplorePage.jsx'), 'utf8');
const defaultDatasetTab = readFileSync(resolve(__dirname, 'DefaultDatasetTab.jsx'), 'utf8');
const myDataTab = readFileSync(resolve(__dirname, 'MyDataTab.jsx'), 'utf8');
const adminReviewPanel = readFileSync(resolve(__dirname, '../../components/AdminReviewPanel.jsx'), 'utf8');

test('ExplorePage exposes three top-level data management sections', () => {
  assert.match(explorePage, /key:\s*'official'/);
  assert.match(explorePage, /key:\s*'personal'/);
  assert.match(explorePage, /key:\s*'admin'/);
  assert.doesNotMatch(explorePage, /key:\s*'governance'/);
  assert.doesNotMatch(explorePage, /<GovernanceTab\s*\/>/);
});

test('Admin Review is role-aware and hidden from non-admin navigation', () => {
  assert.match(explorePage, /isAdmin\s*\?\s*\[/);
  assert.match(explorePage, /AdminReviewWorkspace/);
});

test('official sources show MCD, OpenMARS, and NOMAD role cards', () => {
  assert.match(defaultDatasetTab, /OFFICIAL_SOURCE_CARDS/);
  assert.match(defaultDatasetTab, /key:\s*'mcd'/);
  assert.match(defaultDatasetTab, /key:\s*'openmars'/);
  assert.match(defaultDatasetTab, /key:\s*'nomad'/);
});

test('personal data keeps upload simple and moves dense details behind advanced disclosure', () => {
  assert.match(myDataTab, /advancedOpen/);
  assert.match(myDataTab, /AdvancedDetailSection/);
  assert.match(myDataTab, /copy\.advancedTitle/);
});

test('admin review content is reusable inside the page and in the drawer', () => {
  assert.match(adminReviewPanel, /export function AdminReviewWorkspace/);
  assert.match(adminReviewPanel, /PanelContent/);
});

test('admin review surfaces subscribe to a shared review refresh signal', () => {
  assert.match(adminReviewPanel, /reviewSignal/);
  assert.match(adminReviewPanel, /useEffect\(\(\)\s*=>\s*\{\s*load\(\);?\s*\},\s*\[load,\s*reviewSignal\]\)/);
  assert.match(explorePage, /AdminReviewWorkspace[^>]+reviewSignal=\{reviewSignal\}/);
});

test('personal preview loading ignores stale async responses after selection or collapse changes', () => {
  assert.match(myDataTab, /previewRequestRef/);
  assert.match(myDataTab, /requestId\s*!==\s*previewRequestRef\.current/);
  assert.match(myDataTab, /onSelectDataset/);
});

test('data management chrome uses compact tabs without an explanatory task card', () => {
  assert.match(explorePage, /CompactTaskTabs/);
  assert.doesNotMatch(explorePage, /quickTitle/);
  assert.doesNotMatch(explorePage, /quickDesc/);
  assert.doesNotMatch(explorePage, /<GlowCard[^>]*>\s*<div[^>]*>\s*<div[^>]*>\s*<div[\s\S]*ViewTab/);
});

test('official source page keeps only source cards and a compact coverage summary', () => {
  assert.match(defaultDatasetTab, /OfficialCoverageSummary/);
  assert.doesNotMatch(defaultDatasetTab, /MetricCard/);
  assert.doesNotMatch(defaultDatasetTab, /VariableGroupCard/);
  assert.doesNotMatch(defaultDatasetTab, /OFFICIAL_CAPABILITY_TAG_KEYS/);
  assert.doesNotMatch(defaultDatasetTab, /OFFICIAL_VARIABLE_GROUPS/);
});

test('official source page derives each source status from overview coverage data', () => {
  assert.match(defaultDatasetTab, /fetchOverviewInfo/);
  assert.doesNotMatch(defaultDatasetTab, /fetchDataInfo/);
  assert.match(defaultDatasetTab, /OfficialSourceSituationGrid/);
  assert.match(defaultDatasetTab, /sourceCoverageStats/);
  assert.match(defaultDatasetTab, /coverage\?\.mcd/);
  assert.match(defaultDatasetTab, /coverage\?\.openmars/);
  assert.match(defaultDatasetTab, /coverage\?\.nomad/);
});

test('personal data default page keeps contribution detail secondary', () => {
  assert.match(myDataTab, /PersonalSummaryStrip/);
  assert.match(myDataTab, /summaryStats\.inReview/);
  assert.doesNotMatch(myDataTab, /Usage Boundary/);
  assert.doesNotMatch(myDataTab, /使用边界/);
  assert.doesNotMatch(myDataTab, /copy\.contributionReadyStat}\s+value=\{summaryStats\.contributionReady\}/);
});

test('personal data queue and detail default views stay compact', () => {
  assert.match(myDataTab, /CompactQueueItem/);
  assert.match(myDataTab, /PersonalCoreDetail/);
  assert.doesNotMatch(myDataTab, /copy\.queueDesc/);
  assert.doesNotMatch(myDataTab, /desc=\{copy\.sourceModeDesc\}/);
  assert.doesNotMatch(myDataTab, /\{viewingCtx\.datasetState\.desc\}/);
});

test('personal data distinguishes raw source from generated caches', () => {
  assert.match(myDataTab, /cache_status/);
  assert.match(myDataTab, /raw MCD 3h|原始 MCD 3h/);
  assert.match(myDataTab, /cache_building/);
  assert.match(myDataTab, /mcd_overview/);
  assert.match(myDataTab, /mcd_3h/);
});

test('personal data deletion uses the long-press confirm instead of a modal', () => {
  assert.match(myDataTab, /HoldToDeleteButton/);
  assert.match(myDataTab, /size="compact"/);
  assert.match(myDataTab, /onDelete=\{\(\) => handleDeleteConfirm\(viewingUpload\.id\)\}/);
  assert.doesNotMatch(myDataTab, /ConfirmDialog/);
  assert.doesNotMatch(myDataTab, /confirmDelete/);
});

test('admin inline workspace uses compact panel sizing', () => {
  assert.match(adminReviewPanel, /adminWorkspaceMinHeight/);
  assert.doesNotMatch(adminReviewPanel, /minHeight:\s*640/);
  assert.match(adminReviewPanel, /height:\s*56/);
});
