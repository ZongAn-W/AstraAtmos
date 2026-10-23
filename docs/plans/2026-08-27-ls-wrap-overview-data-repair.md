# Ls Wrap and Overview Data Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct Ls wrap handling and replace the flattened MY27/MY28 overview datasets with validated seasonal data.

**Architecture:** Fix wrap detection at the shared `unwrap_ls` boundary so all interpolation callers receive a monotonic timeline. Prove the defect with focused unit and builder tests, then generate MY27/MY28 through the existing `reference_direct` builder into staging and install only validated artifacts.

**Tech Stack:** Python 3, NumPy, SciPy, Xarray, NetCDF, pytest, PowerShell.

---

## File Map

- Create `AstraAtmos_backend/backend/tests/test_data_align.py`: focused regression coverage for a single Ls wrap.
- Modify `AstraAtmos_backend/backend/tests/test_mcd_overview_builder.py`: regression coverage for overview ozone interpolation across a wrap.
- Modify `AstraAtmos_backend/backend/core/data_align.py`: detect wraps against immutable source values.
- Generate `AstraAtmos_backend/backend/data/mcd_overview_staging_ls_fix/MCD_MY27_overview.nc`: staged MY27 artifact.
- Generate `AstraAtmos_backend/backend/data/mcd_overview_staging_ls_fix/MCD_MY28_overview.nc`: staged MY28 artifact.
- Replace `AstraAtmos_backend/backend/data/mcd_overview/MCD_MY27_overview.nc`: installed generated MY27 data.
- Replace `AstraAtmos_backend/backend/data/mcd_overview/MCD_MY28_overview.nc`: installed generated MY28 data.

The four NetCDF paths are ignored generated data and will not be committed.

### Task 1: Add Failing Ls Wrap Regressions

**Files:**
- Create: `AstraAtmos_backend/backend/tests/test_data_align.py`
- Modify: `AstraAtmos_backend/backend/tests/test_mcd_overview_builder.py`

- [ ] **Step 1: Write the focused helper regression**

Create `test_data_align.py` with the same backend path bootstrap used by other backend tests:

```python
import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.data_align import unwrap_ls  # noqa: E402


def test_unwrap_ls_applies_one_offset_after_one_wrap():
    wrapped = np.array([359.9, 0.1, 0.2, 1.0], dtype=np.float64)

    result = unwrap_ls(wrapped)

    np.testing.assert_allclose(result, [359.9, 360.1, 360.2, 361.0])
    np.testing.assert_allclose(wrapped, [359.9, 0.1, 0.2, 1.0])
```

- [ ] **Step 2: Write the builder interpolation regression**

Add `interpolate_reference_ozone` to the existing import from `build_mcd_overview_dataset`, then add:

```python
def test_interpolate_reference_ozone_preserves_samples_after_ls_wrap():
    reference_ls = np.array([359.5, 0.5, 1.5, 2.5], dtype=np.float32)
    reference_ozone = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(-1, 1, 1)
    target_ls = np.array([0.5, 1.5, 2.5], dtype=np.float32)

    result = interpolate_reference_ozone(reference_ozone, reference_ls, target_ls)

    np.testing.assert_allclose(result[:, 0, 0], [2.0, 3.0, 4.0], rtol=1e-6)
```

- [ ] **Step 3: Run both tests and verify RED**

Run from the repository root:

```powershell
& '<工作区>\venvs\astraatmos-oom-notifications\Scripts\python.exe' -m pytest `
  AstraAtmos_backend/backend/tests/test_data_align.py `
  AstraAtmos_backend/backend/tests/test_mcd_overview_builder.py::test_interpolate_reference_ozone_preserves_samples_after_ls_wrap `
  -q
```

Expected: both tests fail because values after the first wrapped sample receive additional 360-degree offsets and interpolation returns values close to `2.0` instead of `3.0` and `4.0`.

### Task 2: Correct Shared Ls Unwrapping

**Files:**
- Modify: `AstraAtmos_backend/backend/core/data_align.py`
- Test: `AstraAtmos_backend/backend/tests/test_data_align.py`
- Test: `AstraAtmos_backend/backend/tests/test_mcd_overview_builder.py`

- [ ] **Step 1: Implement the minimal source/output separation**

Replace the `unwrap_ls` loop body with:

```python
    source = np.asarray(ls_array, dtype=np.float64)
    unwrapped = source.copy()
    offset = 0.0
    for i in range(1, len(source)):
        if source[i] - source[i - 1] < -180:
            offset += 360.0
        unwrapped[i] += offset
    return unwrapped
```

Wrap detection now compares two unmodified source values, while the output accumulates the offset.

- [ ] **Step 2: Run the focused tests and verify GREEN**

Run:

```powershell
& '<工作区>\venvs\astraatmos-oom-notifications\Scripts\python.exe' -m pytest `
  AstraAtmos_backend/backend/tests/test_data_align.py `
  AstraAtmos_backend/backend/tests/test_mcd_overview_builder.py::test_interpolate_reference_ozone_preserves_samples_after_ls_wrap `
  -q
```

Expected: `2 passed`.

- [ ] **Step 3: Run the complete builder test module**

Run:

```powershell
& '<工作区>\venvs\astraatmos-oom-notifications\Scripts\python.exe' -m pytest `
  AstraAtmos_backend/backend/tests/test_mcd_overview_builder.py -q
```

Expected: zero failures.

- [ ] **Step 4: Commit the code repair**

```powershell
git add -- `
  AstraAtmos_backend/backend/core/data_align.py `
  AstraAtmos_backend/backend/tests/test_data_align.py `
  AstraAtmos_backend/backend/tests/test_mcd_overview_builder.py
git commit -m "fix: preserve Ls samples after year wrap"
```

### Task 3: Rebuild and Validate MY27/MY28 in Staging

**Files:**
- Read: `AstraAtmos_backend/backend/data/MCD_Output_global_10m_ls_lst/MCD_MY27_global_3h_5deg_10m_ls_lst.nc`
- Read: `AstraAtmos_backend/backend/data/MCD_Output_global_10m_ls_lst/MCD_MY28_global_3h_5deg_10m_ls_lst.nc`
- Generate: `AstraAtmos_backend/backend/data/mcd_overview_staging_ls_fix/MCD_MY27_overview.nc`
- Generate: `AstraAtmos_backend/backend/data/mcd_overview_staging_ls_fix/MCD_MY28_overview.nc`

- [ ] **Step 1: Generate both staged files with the direct reference builder**

Run:

```powershell
& '<工作区>\venvs\astraatmos-oom-notifications\Scripts\python.exe' `
  AstraAtmos_backend/backend/scripts/build_mcd_overview_dataset.py `
  --years 27 28 `
  --reference-dir AstraAtmos_backend/backend/data/MCD_Output_global_10m_ls_lst `
  --output-dir AstraAtmos_backend/backend/data/mcd_overview_staging_ls_fix `
  --mode reference
```

Expected: exactly one `Wrote` line for MY27 and one for MY28.

- [ ] **Step 2: Validate staged schemas, coordinates, units, and seasonal ranges**

Run this read-only validator from the repository root:

```powershell
@'
from pathlib import Path

import numpy as np
import xarray as xr

root = Path("AstraAtmos_backend/backend/data/mcd_overview_staging_ls_fix")
required = {
    "Ls", "o3col", "Pressure", "Temperature", "U_Wind", "V_Wind",
    "Dust_Optical_Depth", "Solar_Flux_DN", "lat", "lon",
}

for year in (27, 28):
    path = root / f"MCD_MY{year}_overview.nc"
    with xr.open_dataset(path, decode_times=False) as ds:
        assert required.issubset(set(ds.variables)), (year, sorted(required - set(ds.variables)))
        assert ds.attrs.get("build_mode") == "reference_direct"
        ls = np.asarray(ds["Ls"].values, dtype=np.float64)
        assert np.isfinite(ls).all()
        assert np.all(np.diff(ls) > 0)
        assert np.unique(ls).size == ls.size
        assert float(ls.min()) >= 0.0 and float(ls.max()) < 360.0
        assert ds["o3col"].attrs.get("units") == "um-atm"
        ozone = np.asarray(ds["o3col"].values, dtype=np.float64)
        zonal = np.nanmean(ozone, axis=2)
        seasonal_range = np.nanmax(zonal, axis=0) - np.nanmin(zonal, axis=0)
        median_range = float(np.nanmedian(seasonal_range))
        assert median_range > 1.0, (year, median_range)
        temperature = np.asarray(ds["Temperature"].values, dtype=np.float64)
        solar = np.asarray(ds["Solar_Flux_DN"].values, dtype=np.float64)
        assert float(np.nanstd(np.nanmean(temperature, axis=(1, 2)))) > 1.0
        assert float(np.nanstd(np.nanmean(solar, axis=(1, 2)))) > 1.0
        print(year, dict(ds.sizes), f"median_seasonal_range={median_range:.4f} um-atm")
'@ | & '<工作区>\venvs\astraatmos-oom-notifications\Scripts\python.exe' -
```

Expected: MY27 and MY28 print valid dimensions and median seasonal ranges above `1.0 um-atm`. If either assertion fails, stop before replacing installed files.

### Task 4: Install Artifacts and Verify the Overview Contract

**Files:**
- Replace: `AstraAtmos_backend/backend/data/mcd_overview/MCD_MY27_overview.nc`
- Replace: `AstraAtmos_backend/backend/data/mcd_overview/MCD_MY28_overview.nc`

- [ ] **Step 1: Resolve and verify the four exact paths**

Run:

```powershell
$repoRoot = (Resolve-Path '.').Path
$stage27 = (Resolve-Path 'AstraAtmos_backend/backend/data/mcd_overview_staging_ls_fix/MCD_MY27_overview.nc').Path
$stage28 = (Resolve-Path 'AstraAtmos_backend/backend/data/mcd_overview_staging_ls_fix/MCD_MY28_overview.nc').Path
$target27 = [System.IO.Path]::GetFullPath((Join-Path $repoRoot 'AstraAtmos_backend/backend/data/mcd_overview/MCD_MY27_overview.nc'))
$target28 = [System.IO.Path]::GetFullPath((Join-Path $repoRoot 'AstraAtmos_backend/backend/data/mcd_overview/MCD_MY28_overview.nc'))
foreach ($path in @($stage27, $stage28, $target27, $target28)) {
    if (-not $path.StartsWith($repoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside repository: $path"
    }
    Write-Output $path
}
```

Expected: all four absolute paths are inside `AstraAtmos`.

- [ ] **Step 2: Replace each installed artifact explicitly**

Run one move per exact file:

```powershell
Move-Item `
  -LiteralPath 'AstraAtmos_backend\backend\data\mcd_overview_staging_ls_fix\MCD_MY27_overview.nc' `
  -Destination 'AstraAtmos_backend\backend\data\mcd_overview\MCD_MY27_overview.nc' `
  -Force
Move-Item `
  -LiteralPath 'AstraAtmos_backend\backend\data\mcd_overview_staging_ls_fix\MCD_MY28_overview.nc' `
  -Destination 'AstraAtmos_backend\backend\data\mcd_overview\MCD_MY28_overview.nc' `
  -Force
```

No wildcard, recursive deletion, or directory deletion is permitted.

- [ ] **Step 3: Verify the installed files through `AnalysisService`**

Run:

```powershell
@'
from pathlib import Path
import sys

import numpy as np
import xarray as xr

backend = Path("AstraAtmos_backend/backend").resolve()
sys.path.insert(0, str(backend))
from services.analysis_service import AnalysisService


class OverviewView:
    def __init__(self, path):
        with xr.open_dataset(path, decode_times=False) as ds:
            self.data = {
                "o3col": np.asarray(ds["o3col"].values, dtype=np.float32),
                "ls": np.asarray(ds["Ls"].values, dtype=np.float32),
                "lat": np.asarray(ds["lat"].values, dtype=np.float32),
                "lon": np.asarray(ds["lon"].values, dtype=np.float32),
            }

    def get_openmars_data(self, mars_year):
        return self.data


root = backend / "data" / "mcd_overview"
for year in (27, 28):
    service = AnalysisService(OverviewView(root / f"MCD_MY{year}_overview.nc"))
    payload = service.get_seasonal_heatmap(year)
    matrix = np.asarray(payload["z"], dtype=np.float64)
    latitude_ranges = np.nanmax(matrix, axis=1) - np.nanmin(matrix, axis=1)
    median_range = float(np.nanmedian(latitude_ranges))
    assert matrix.shape[0] == len(payload["y"])
    assert matrix.shape[1] == len(payload["x"])
    assert median_range > 1.0, (year, median_range)
    print(year, matrix.shape, f"api_median_seasonal_range={median_range:.4f} um-atm")
'@ | & '<工作区>\venvs\astraatmos-oom-notifications\Scripts\python.exe' -
```

Expected: both service matrices have matching axes and a median seasonal range above `1.0 um-atm`.

- [ ] **Step 4: Run related backend regression tests**

Run:

```powershell
& '<工作区>\venvs\astraatmos-oom-notifications\Scripts\python.exe' -m pytest `
  AstraAtmos_backend/backend/tests/test_data_align.py `
  AstraAtmos_backend/backend/tests/test_mcd_overview_builder.py `
  AstraAtmos_backend/backend/tests/test_mcd_overview_service.py `
  AstraAtmos_backend/backend/tests/test_analysis_overview_routes.py `
  AstraAtmos_backend/backend/tests/test_analysis_point_probe.py `
  -q
```

Expected: zero failures.

- [ ] **Step 5: Check repository scope**

Run:

```powershell
git diff --check
git status --short
git log -3 --oneline
```

Expected: no whitespace errors, no uncommitted source changes, and only the planned documentation and code commits at the top of history. Generated NetCDF files remain ignored.

- [ ] **Step 6: Restart the backend service if it is running**

The app caches loaded overview files and heatmap responses. Restart the existing backend process through its normal launcher so MY27/MY28 are reloaded. Do not terminate unrelated Python processes.
