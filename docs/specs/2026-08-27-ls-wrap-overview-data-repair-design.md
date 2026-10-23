# Ls Wrap and Overview Data Repair Design

## Problem

The Data Overview seasonal ozone heatmap for the default MY27 source is nearly constant along the Ls axis. The frontend renders the matrix returned by the backend correctly; the seasonal signal is already flattened in `MCD_MY27_overview.nc` and `MCD_MY28_overview.nc`.

The shared `unwrap_ls` implementation compares each raw current value with an already adjusted previous value. After the first `359 -> 0` wrap, every following sample is treated as another wrap. For example, `[359.9, 0.1, 0.2]` becomes `[359.9, 360.1, 720.2]`. The legacy overview builder then interpolates almost the full target year between adjacent source samples.

## Scope

This repair will:

- Correct the shared Ls unwrapping behavior.
- Add regression coverage for the shared helper and overview ozone interpolation.
- Rebuild the official MY27 and MY28 overview datasets from their canonical reference MCD files.
- Validate the rebuilt files and the resulting seasonal heatmap matrix.

It will not change chart layout, color scales, API schemas, unrelated variables, uploaded datasets, or other Mars-year artifacts.

## Code Design

`unwrap_ls` will preserve an immutable copy of the original wrapped input for transition detection. The output will be a separate floating-point array. A negative transition greater than 180 degrees in the original sequence increments the cumulative offset once; later samples retain that offset until another genuine original wrap occurs.

The public function signature and return type remain unchanged, so existing callers in MCD/OpenMARS alignment and overview construction require no changes.

## Test Design

Tests will be written before the production change and observed failing for the current defect.

1. A focused `unwrap_ls` test will assert that values following one wrap receive one cumulative 360-degree offset, not an additional offset per sample.
2. An overview interpolation test will use a small chronological reference sequence crossing `359 -> 0` and assert that target Ls values map to their corresponding source fields.
3. Existing overview builder, analysis route, and data alignment tests will run after the focused tests pass.

## Data Rebuild

MY27 and MY28 will be generated into a dedicated staging directory using the current `reference_direct` path. This keeps ozone and environmental variables on the same raw MCD timeline and matches the format used by the unaffected MY24-MY26 and MY29-MY35 artifacts.

The current files remain untouched until both staged files pass validation. After validation, only these exact targets will be replaced:

- `AstraAtmos_backend/backend/data/mcd_overview/MCD_MY27_overview.nc`
- `AstraAtmos_backend/backend/data/mcd_overview/MCD_MY28_overview.nc`

These generated files are ignored by Git and can be reproduced from the canonical raw MCD inputs.

## Validation

Each staged and installed file must satisfy all of the following:

- Required dimensions and fields are present and internally consistent.
- Ls values are unique, sorted, finite, and remain in `[0, 360)`.
- Ozone units load as `um-atm`.
- The median latitude-wise seasonal ozone range exceeds `1.0 um-atm`; the defective files are below `0.03 um-atm` and the raw sources are near `7 um-atm`.
- Temperature and solar-flux fields retain physically non-constant seasonal variation.
- `AnalysisService.get_seasonal_heatmap` returns a matrix whose Ls dimension varies materially.

The backend service must be restarted after replacement because overview datasets and heatmap responses are cached in memory.

## Error Handling

If generation or validation fails for either year, no installed overview file will be replaced. If post-install verification fails, the raw reference files remain available for a clean rebuild; no unrelated generated data will be removed.
