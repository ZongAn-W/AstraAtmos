# Shared Mars and Earth Overview Visual Shell Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Make the Earth and Mars data-overview pages use the same panel layout, visual language, background, and 3D particle-globe presentation while preserving each planet’s existing data semantics, APIs, analysis cards, and interactions.

**Architecture:** Move Mars from its legacy external-panel positioning into the shared OverviewShell slots layout already used by Earth. Put widths, surface colors, borders, blur, spacing, responsive breakpoints, timeline placement, legend placement, and Copilot offsets under one shared visual contract. Keep Mars and Earth adapters/renderers responsible for their own data, units, geometry, lighting, texture, and capability declarations; only the shell and renderer primitives become shared.

**Tech Stack:** React 19, Vite 6, Three.js, OverviewShell, SphericalFieldCanvas, OverviewScene, CSS custom properties, Node.js built-in test runner.

---

## Scope and acceptance criteria

The implementation is visual and structural. It must not change these contracts:

- Mars continues to use MY/Ls, MCD/OpenMARS/NOMAD source selection, uploaded-source filtering, point probing, gesture controls, seasonal sunlight, and all existing Mars analysis requests.
- Earth continues to use ISO dates, the Earth MERRA-2 v2 fingerprint, DU/K/m s⁻¹/W m⁻² units, global 36×72 geometry, area-weighted means, Earth research endpoints, and the daily-mean limitation for the diurnal card.
- Switching Mars and Earth still cancels in-flight requests, clears stale field/point state, resets camera identity, and restores each planet’s own selection.
- Desktop pages have the same left control rail, central scene, right analysis rail, bottom timeline, legend, and Copilot anchor. The rails use the same width defaults and drag behavior.
- Both planets use the same dark-space visual treatment and particle-field conventions. Mars keeps its Mars texture and seasonal light; Earth keeps its coastline/base-map overlay and fixed Earth lighting.
- At the compact breakpoint both planets use the same stacked layout and do not create horizontal overflow.

## File map

- frontend/src/pages/DataOverviewPage/workbench/OverviewShell.jsx: shared desktop/compact slots, rail widths, resize handles, and scene/timeline/overlay composition.
- frontend/src/pages/DataOverviewPage/workbench/overviewWorkbench.css: shared shell geometry and Plotly mode-bar styling.
- frontend/src/pages/DataOverviewPage.jsx: Mars page composition, loading, point probing, gesture handling, and source requests.
- frontend/src/pages/DataOverviewPage/SidebarMenu.jsx: Mars controls and source selectors; currently owns a fixed left rail.
- frontend/src/pages/DataOverviewPage/DetailPanel.jsx: Mars analysis-card mounting; currently owns a fixed right rail.
- frontend/src/pages/DataOverviewPage/TimelineController.jsx: Mars Ls timeline and source-coverage display.
- frontend/src/pages/DataOverviewPage/GlobeLegend.jsx: Mars field/source legend.
- frontend/src/pages/DataOverviewPage/AICopilotWidget.jsx: Mars Copilot surface and anchor.
- frontend/src/pages/DataOverviewPage/Mars3DBackground.jsx: Mars texture, lighting, and particle scene.
- frontend/src/pages/DataOverviewPage/EarthOverview/EarthWorkbenchScene.jsx: Earth shared-shell composition and Earth-specific controls.
- frontend/src/pages/DataOverviewPage/workbench/OverviewScene.jsx: shared scene adapter boundary.
- frontend/src/components/SphericalFieldCanvas.jsx: shared particle globe, camera, and picking.
- frontend/src/components/sphericalRegionalGrid.js: Earth grid geometry and field-color mapping.
- frontend/src/index.css: global theme variables and possible legacy overview conflicts.
- frontend/src/contexts/DataOverviewContext.jsx: Mars rail-width state.
- frontend/src/pages/DataOverviewPage/workbench/*.test.js, frontend/src/pages/DataOverviewPage/*.test.js, frontend/src/components/*test.js: regression coverage.
- README.md and docs/earth-analysis-workbench.md: shared workbench architecture documentation.

### Task 1: Establish a visual contract and baseline tests

Files:
- Create frontend/src/pages/DataOverviewPage/workbench/overviewVisualContract.js
- Create frontend/src/pages/DataOverviewPage/workbench/overviewVisualContract.test.js
- Modify frontend/src/pages/DataOverviewPage/workbench/OverviewShell.jsx

- [ ] Step 1: Define shared layout and surface constants.

Create a pure module containing the values consumed by the shell and tests:

~~~js
export const OVERVIEW_LAYOUT = Object.freeze({
  navbarHeight: 70,
  leftWidth: 300,
  rightWidth: 540,
  minSceneWidth: 320,
  compactBreakpoint: 1120,
  timelineBottom: 18,
});

export const OVERVIEW_SURFACE = Object.freeze({
  darkPanel: 'rgba(10,12,18,0.62)',
  darkPanelStrong: 'rgba(10,12,18,0.82)',
  darkBorder: 'rgba(255,255,255,0.08)',
  lightPanel: 'rgba(255,255,255,0.86)',
  lightBorder: 'rgba(15,23,42,0.10)',
  blur: '20px',
});

export function desktopSceneWidth(viewportWidth, left = OVERVIEW_LAYOUT.leftWidth, right = OVERVIEW_LAYOUT.rightWidth) {
  return viewportWidth - left - right;
}

export function shouldUseCompactOverview(viewportWidth) {
  return viewportWidth <= OVERVIEW_LAYOUT.compactBreakpoint;
}
~~~

- [ ] Step 2: Test the contract.

Cover default widths, the scene-width calculation, the compact breakpoint, and equal rail defaults:

~~~powershell
cd frontend
node --test src/pages/DataOverviewPage/workbench/overviewVisualContract.test.js
~~~

Expected: PASS after the module is created.

- [ ] Step 3: Make OverviewShell consume the contract.

Replace hard-coded navbar and minimum-scene values with imports. Replace the current width <= 900 check with shouldUseCompactOverview while retaining both slots and external modes temporarily.

- [ ] Step 4: Run focused shell tests.

~~~powershell
node --test src/pages/DataOverviewPage/workbench/overviewVisualContract.test.js src/pages/DataOverviewPage/workbench/overviewShellStructure.test.js
~~~

Expected: PASS.

### Task 2: Centralize shell visual tokens

Files:
- Modify frontend/src/pages/DataOverviewPage/workbench/overviewWorkbench.css
- Modify frontend/src/index.css
- Modify frontend/src/pages/DataOverviewPage/workbench/OverviewShell.jsx
- Test frontend/src/pages/DataOverviewPage/workbench/overviewShellStructure.test.js

- [ ] Step 1: Add shared CSS custom properties.

Add these properties to .overview-shell and switch the surface values in .overview-shell.is-light:

~~~css
.overview-shell {
  --overview-left-width: 300px;
  --overview-right-width: 540px;
  --overview-navbar-height: 70px;
  --overview-panel-bg: rgba(10, 12, 18, 0.62);
  --overview-panel-bg-strong: rgba(10, 12, 18, 0.82);
  --overview-panel-border: rgba(255, 255, 255, 0.08);
  --overview-panel-blur: 20px;
  --overview-card-radius: 16px;
  --overview-control-radius: 12px;
  --overview-accent: #ff6b35;
  background: var(--bg-page, #060a12);
}

.overview-shell.is-light {
  --overview-panel-bg: rgba(255, 255, 255, 0.86);
  --overview-panel-bg-strong: rgba(255, 255, 255, 0.94);
  --overview-panel-border: rgba(15, 23, 42, 0.10);
  background: var(--bg-page, #f4f7fb);
}
~~~

Add shared classes for panel surfaces, section labels, controls, timeline, legend, and overlay anchors. Keep Earth variable colors and Mars source colors outside the shared token block.

- [ ] Step 2: Remove conflicting overview backgrounds.

Search index.css for .space-scene, .earth-scene, .overview-shell, body, and fixed overlay rules. Keep global typography and app variables. Move overview backgrounds and overview z-index rules into overviewWorkbench.css; keep Earth map rules under .earth-scene and Mars texture rules under .mars-scene.

- [ ] Step 3: Tokenize OverviewShell panels.

Replace inline panelBackground and borderSoft values with shared CSS variables/classes. Keep both accessible resize separators and their mouse behavior unchanged.

- [ ] Step 4: Extend structure tests.

Assert shared CSS import, panel variables/classes, compact helper usage, and both resize separators. Run the focused test and npm run build.

### Task 3: Migrate Mars into shared slots

Files:
- Modify frontend/src/pages/DataOverviewPage.jsx
- Modify frontend/src/pages/DataOverviewPage/SidebarMenu.jsx
- Modify frontend/src/pages/DataOverviewPage/DetailPanel.jsx
- Modify frontend/src/contexts/DataOverviewContext.jsx
- Modify frontend/src/pages/DataOverviewPage/workbench/OverviewShell.jsx
- Test frontend/src/pages/DataOverviewPage/overviewSceneModel.test.js
- Test frontend/src/pages/DataOverviewPage/overviewOverlayHudStructure.test.js

- [ ] Step 1: Add embedded mode to SidebarMenu.

When embedded is true, retain all controls, source requests, state updates, collapsible sections, and labels, but omit the fixed left wrapper and component-owned resize handle. Use the shell’s padding and overflow. Keep the old wrapper only until the page switch is verified.

- [ ] Step 2: Add embedded mode to DetailPanel.

When embedded is true, retain OverviewAnalysisPanel, lazy card mounting, selected-coordinate reset, card state, and resize-trigger effects, but omit fixed right positioning, slide-in transition, width, and component-owned resize handle. The shell becomes the sole owner of rail geometry.

- [ ] Step 3: Mount Mars through OverviewShell slots.

Use this composition in DataOverviewPage.jsx:

~~~jsx
<OverviewShell
  planet="mars"
  isLight={isLight}
  leftWidth={leftPanelWidth}
  rightWidth={rightPanelWidth}
  onLeftWidthChange={setLeftPanelWidth}
  onRightWidthChange={setRightPanelWidth}
  panelMode="slots"
  scene={marsScene}
  sidebar={<SidebarMenu embedded sceneSwitch={sceneSwitch} />}
  analysis={<DetailPanel embedded sliceData={mcdMainSlice} overviewSourceParams={overviewSourceParams} />}
  timeline={<TimelineController embedded />}
  overlay={marsOverlay}
  notification={notification}
/>
~~~

Keep Mars loading, request effects, point probing, gestures, scene model, and source semantics unchanged.

- [ ] Step 4: Remove duplicate positioning.

After browser parity is confirmed, delete the Mars compatibility wrapper and its unused fixed-position constants, visibility transition, panel background calculations, and resize handlers. Leave one source of truth for widths.

- [ ] Step 5: Run Mars regression tests.

~~~powershell
cd frontend
node --test src/pages/DataOverviewPage/pointProbeStructure.test.js src/pages/DataOverviewPage/overviewOverlayHudStructure.test.js src/pages/DataOverviewPage/overviewSolarLightingStructure.test.js src/pages/DataOverviewPage/workbench/overviewShellStructure.test.js
~~~

Expected: PASS, with no API or request-coordinator changes.

### Task 4: Unify timeline, legend, Copilot, and overlay anchors

Files:
- Modify frontend/src/pages/DataOverviewPage/TimelineController.jsx
- Modify frontend/src/pages/DataOverviewPage/GlobeLegend.jsx
- Modify frontend/src/pages/DataOverviewPage/AICopilotWidget.jsx
- Modify frontend/src/pages/DataOverviewPage/EarthOverview/EarthWorkbenchScene.jsx
- Modify frontend/src/pages/DataOverviewPage.jsx
- Modify frontend/src/pages/DataOverviewPage/workbench/overviewWorkbench.css

- [ ] Step 1: Add shell anchor variables.

Expose CSS variables for scene-left, scene-right, timeline-bottom, and overlay-gap. Set them from the shell’s actual widths so rail dragging updates every overlay without each child calculating viewport offsets.

- [ ] Step 2: Convert Mars timeline positioning.

Add embedded mode to TimelineController and use the shared anchor. Keep Mars source-coverage segments, Ls labels, playback, and source dots unchanged. Keep EarthTimeline date semantics unchanged and place it in the same shell slot.

- [ ] Step 3: Convert both legends.

Keep Mars multi-source/validation/difference content and Earth physical-unit content. Unify position, surface, padding, typography, and collision behavior.

- [ ] Step 4: Convert Copilot positioning.

Keep Mars AI payload and Earth digest-only payload unchanged. Use the shared right-rail offset and move the bubble above the timeline or into the analysis column when compact.

- [ ] Step 5: Add overlay tests.

Assert timeline, legend, and Copilot reference shared shell anchors instead of independent viewport offsets.

### Task 5: Align the 3D particle globe presentation

Files:
- Modify frontend/src/components/SphericalFieldCanvas.jsx
- Modify frontend/src/pages/DataOverviewPage/workbench/OverviewScene.jsx
- Modify frontend/src/pages/DataOverviewPage/Mars3DBackground.jsx
- Modify frontend/src/components/sphericalRegionalGrid.js
- Modify frontend/src/index.css
- Test frontend/src/components/SphericalFieldCanvasStructure.test.js
- Test frontend/src/components/sphericalRegionalGrid.test.js

- [ ] Step 1: Define explicit shared presentation props.

Use particlePalette, particleDensity, particleSize, globeMaterial, showBaseMap, lightingMode, and planet props. Both scenes pass these explicitly. Keep existing picking, camera-pose, and field-normalization behavior.

- [ ] Step 2: Align visual values.

Use Mars’s dark navy globe, orange/blue highlights, particle density, opacity, and grid-line strength as shared defaults. Earth adds coastline and fixed lighting; Mars adds texture and seasonal light. Do not reuse Mars unit conversion or Ls logic in Earth.

- [ ] Step 3: Preserve planet-specific layers.

Keep Earth coastline and 36×72 geometry in sphericalEarthBaseMap.js and sphericalRegionalGrid.js. Keep Mars texture, geo annotations, and solar-light direction in Mars3DBackground.jsx. The common layer owns material/particle styling only.

- [ ] Step 4: Test renderer props.

Assert both scenes pass planet, field, geometry, shared presentation props, and explicit lighting/base-map flags. Keep all existing seam, pole, geometry, and picking tests.

### Task 6: Normalize controls and analysis cards

Files:
- Modify frontend/src/pages/DataOverviewPage/SidebarMenu.jsx
- Modify frontend/src/pages/DataOverviewPage/DetailPanel.jsx
- Modify frontend/src/pages/DataOverviewPage/EarthOverview/EarthWorkbenchScene.jsx
- Modify frontend/src/pages/DataOverviewPage/EarthOverview/EarthOverviewScene.jsx
- Modify frontend/src/pages/DataOverviewPage/workbench/OverviewCard.jsx
- Modify frontend/src/pages/DataOverviewPage/workbench/OverviewSidebarParts.jsx
- Modify frontend/src/pages/DataOverviewPage/workbench/OverviewAnalysisPanel.jsx
- Modify frontend/src/pages/DataOverviewPage/EarthOverview/earthOverview.css

- [ ] Step 1: Replace duplicated inline surfaces.

Use shared classes for control cards, segmented buttons, selects, switches, mode cards, section labels, and card headers. Keep labels, disabled states, bindings, card catalogs, and handlers unchanged.

- [ ] Step 2: Match Earth control dimensions.

Match Mars padding, control height, radius, font scale, focus ring, and disabled opacity. Earth coordinate inputs, date selectors, variable selectors, and units remain Earth-specific.

- [ ] Step 3: Match analysis-card surfaces.

Use the same GlowCard surface, title row, expand animation, loading/error/unsupported body, Plotly mode bar, and scrollbar treatment. Mars still mounts existing chart components; Earth still renders EarthResearchViews and the daily-mean unsupported card.

- [ ] Step 4: Verify light theme.

Only surface tokens and contrast change under settings.theme === light. Earth CSS must not reset the shared light surface.

### Task 7: Browser-level visual and interaction verification

Files:
- Modify only the implementation files above when a verification failure identifies a concrete regression.

- [ ] Step 1: Verify services and data before judging loading.

Use the existing healthy services or start them from the documented directories. Confirm /health returns {"status":"healthy"} and the Earth v2 context returns HTTP 200 with 36 × 72 geometry.

- [ ] Step 2: Verify Mars at desktop width.

At least 1280px wide, confirm identical slots, rail widths, separators, timeline, legend, and Copilot anchors. Drag both separators. Exercise source toggles, MY selector, variables, point probe, card expansion, gesture toggle, playback, Mars texture, and seasonal light.

- [ ] Step 3: Verify Earth at desktop width.

Confirm identical rail surfaces and spacing. Confirm Earth still shows 2020-01-01, 731 days, 36 × 72, DU, and the area-weighted mean after loading.

- [ ] Step 4: Verify cross-planet isolation.

Switch Earth → Mars → Earth during field and card loading. Confirm no cross-planet requests, no stale field/point series, and independent selection/camera restoration.

- [ ] Step 5: Verify compact and light layouts.

Check 1120px, 900px, and 390px for both planets. Confirm identical stacking, no horizontal scrollbar, reachable timeline, usable globe height, and correct light-theme contrast.

### Task 8: Full validation and documentation synchronization

Files:
- Modify README.md if its shell architecture wording becomes inaccurate.
- Modify docs/earth-analysis-workbench.md if it still describes Mars as an external compatibility path.
- Modify docs/earth-overview.md only when a user-visible Earth label or control changes.

- [ ] Step 1: Run frontend tests and build.

~~~powershell
cd frontend
node --test
npm run build
~~~

Expected: all collected tests pass and Vite completes a production build. A pre-existing chunk-size warning is informational.

- [ ] Step 2: Run Earth backend regression tests.

~~~powershell
cd AstraAtmos_backend\backend
python -m pytest tests/test_earth_overview_service.py tests/test_earth_overview_routes.py tests/test_earth_research_service.py tests/test_earth_research_routes.py
~~~

Expected: PASS. No backend source or schema changes are expected for the visual refactor.

- [ ] Step 3: Check diff and links.

~~~powershell
cd AstraAtmos
git diff --check
git status --short
~~~

If Mars is fully mounted through shared slots, update the README/module wording to describe the actual shared-shell behavior while keeping all functional boundaries and Earth limitations.

- [ ] Step 4: Commit in reviewable units.

Use separate commits for shared visual contract/tokens, Mars slot migration/overlay anchors, 3D presentation alignment, and control/card styling/documentation. Each commit must pass its focused tests. Do not include generated frontend/dist files unless the release workflow requires them.

## Self-review checklist

- The plan covers panel layout, sizes, colors, styles, background, particle globe, responsive behavior, and both themes.
- Existing data/API behavior remains owned by the existing Mars/Earth adapters and page logic.
- No step changes units, time models, capabilities, request paths, or cache identity.
- Every shared value has one source of truth and a focused test.
- Browser verification covers loading, switching, resizing, compact layout, and stale-request isolation.
- README and Earth workbench documentation are updated only if their architecture wording becomes inaccurate.
