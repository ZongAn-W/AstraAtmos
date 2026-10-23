# Uploaded Model MOLA Topography Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit, coordinate-aligned MOLA topography input to uploaded models while preserving all current single-input and Ls-input behavior.

**Architecture:** A shared uploaded-model contract performs strict metadata-driven four-way dispatch. A separate MOLA module validates the packaged 36 x 72 NASA/PDS asset and resamples it by target coordinates; data loaders preserve the real NetCDF grid, while training and inference reuse one static `[1, H, W]` tensor and expand it per batch.

**Tech Stack:** Python 3, PyTorch, NumPy, SciPy, NetCDF4, pytest.

---

## File Map

- Modify `AstraAtmos_backend/backend/config.py`: canonical MOLA asset path with an environment override.
- Modify `AstraAtmos_backend/backend/training_backbones/uploaded_model_contract.py`: strict `ls`/`topography` metadata, validation, batch expansion, and four-way dispatch.
- Create `AstraAtmos_backend/backend/training_backbones/mola_topography.py`: asset validation, target-grid validation, periodic resampling, and static tensor preparation.
- Modify `AstraAtmos_backend/backend/training_backbones/user_model_runner.py`: preserve target coordinates and reuse static topography during all training phases.
- Modify `AstraAtmos_backend/backend/services/user_model_validator.py`: real-MOLA topography dry-run.
- Modify `AstraAtmos_backend/backend/services/inference_service.py`: formal prediction, test metrics/results, and permutation importance propagation.
- Create `AstraAtmos_backend/backend/scripts/build_mola_topography_asset.py`: reproducible PDS download, verification, 5-degree aggregation, NetCDF generation, and validation mode.
- Create `AstraAtmos_backend/backend/data/assets/mola_topography_5deg.nc`: verified platform asset.
- Create `AstraAtmos_backend/backend/tests/test_mola_topography.py`: resampling and asset validation tests.
- Modify `AstraAtmos_backend/backend/tests/test_uploaded_model_runner.py`: contract, grid propagation, training, batch expansion, and compatibility tests.
- Modify `AstraAtmos_backend/backend/tests/test_user_model_validator.py`: topography-only and combined dry-run tests.
- Modify `AstraAtmos_backend/backend/tests/test_uploaded_model_ls_inference.py`: all inference path and permutation-isolation tests, renamed only if useful during implementation.
- Modify `AstraAtmos_backend/backend/tests/test_user_model_downloads.py`: template/guide contract tests.
- Modify `docs/uploaded-model-template.py`: comment-only topography opt-in examples.
- Modify `docs/uploaded-model-training.md`: public topography contract and behavior.
- Create `docs/mola-topography-asset.md`: source product, checksum, resolution, and preprocessing record.

### Task 1: Extend The Shared Uploaded-Model Contract

**Files:**
- Modify: `AstraAtmos_backend/backend/training_backbones/uploaded_model_contract.py`
- Modify: `AstraAtmos_backend/backend/tests/test_uploaded_model_runner.py`

- [ ] **Step 1: Write failing metadata and dispatch tests**

Add canonical specs and recording models to `test_uploaded_model_runner.py`:

```python
TOPOGRAPHY_SPEC = {
    "name": "TopographyModel",
    "auxiliary_inputs": {
        "topography": {
            "required": True,
            "shape": ["batch", 1, "height", "width"],
            "dtype": "float32",
            "unit": "meter",
        }
    },
}
LS_TOPOGRAPHY_SPEC = {
    "name": "LsTopographyModel",
    "auxiliary_inputs": {
        **LS_MODEL_SPEC["auxiliary_inputs"],
        **TOPOGRAPHY_SPEC["auxiliary_inputs"],
    },
}

class _TopographyRecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.arguments = None

    def forward(self, x, topography):
        self.arguments = (x, topography)
        return x[:, -1:, :1]

class _LsTopographyRecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.arguments = None

    def forward(self, x, ls, topography):
        self.arguments = (x, ls, topography)
        return x[:, -1:, :1]
```

Add tests proving:

```python
def test_topography_model_receives_only_x_and_topography():
    model = attach_uploaded_model_contract(_TopographyRecordingModel(), TOPOGRAPHY_SPEC)
    x = torch.zeros(2, 3, 1, 8, 16)
    topo = torch.ones(2, 1, 8, 16, dtype=torch.float32)
    run_uploaded_model(model, x, ls=torch.ones(2, 3), topography=topo)
    assert model.arguments[0] is x
    assert model.arguments[1] is topo

def test_combined_model_argument_order_is_x_ls_topography():
    model = attach_uploaded_model_contract(_LsTopographyRecordingModel(), LS_TOPOGRAPHY_SPEC)
    x = torch.zeros(2, 3, 1, 8, 16)
    ls = torch.ones(2, 3)
    topo = torch.ones(2, 1, 8, 16)
    run_uploaded_model(model, x, ls=ls, topography=topo)
    assert model.arguments[0] is x
    assert model.arguments[1] is ls
    assert model.arguments[2] is topo

def test_undeclared_topography_is_not_passed_to_legacy_or_ls_models():
    x = torch.zeros(2, 3, 1, 8, 16)
    topo = torch.ones(2, 1, 8, 16)
    legacy = attach_uploaded_model_contract(_LegacyRecordingModel(), {"name": "Legacy"})
    ls_model = attach_uploaded_model_contract(_LsRecordingModel(), LS_MODEL_SPEC)
    run_uploaded_model(legacy, x, topography=topo)
    run_uploaded_model(ls_model, x, ls=torch.ones(2, 3), topography=topo)
    assert legacy.argument_count == 1
    assert ls_model.received_ls.shape == (2, 3)
```

Parameterize invalid topography declarations for wrong `required`, `shape`,
`dtype`, `unit`, missing/extra fields, and an unknown third auxiliary name.
Parameterize runtime values `None`, list, integer tensor, rank 3, wrong channel,
wrong batch, wrong height/width, NaN, and infinity. Add an expansion test:

```python
def test_static_topography_expands_without_copy_for_changing_batch_size():
    grid = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    batch = expand_topography_batch(torch.zeros(3, 5, 1, 4, 8), grid, "expand test")
    assert batch.shape == (3, 1, 4, 8)
    assert batch.dtype == torch.float32
    assert batch.stride(0) == 0
    assert batch.untyped_storage().data_ptr() == grid.untyped_storage().data_ptr()
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m pytest AstraAtmos_backend/backend/tests/test_uploaded_model_runner.py -k "topography or auxiliary_input_metadata" -q
```

Expected: collection or assertions fail because topography metadata,
`validate_topography_tensor`, and `expand_topography_batch` do not exist.

- [ ] **Step 3: Implement strict metadata and runtime validation**

In `uploaded_model_contract.py`, add:

```python
TOPOGRAPHY_AUXILIARY_INPUT = {
    "required": True,
    "shape": ["batch", 1, "height", "width"],
    "dtype": "float32",
    "unit": "meter",
}
AUXILIARY_INPUT_SPECS = {
    "ls": LS_AUXILIARY_INPUT,
    "topography": TOPOGRAPHY_AUXILIARY_INPUT,
}
```

Replace the ls-only normalizer with a loop over the declared subset. Reject
keys outside `AUXILIARY_INPUT_SPECS`; for each allowed entry, retain the current
field-level missing, extra, type, expected, and actual error format. Return a
new canonical dict in fixed `ls`, then `topography` order.

Add:

```python
def uploaded_model_requires_topography(model: nn.Module) -> bool:
    return "topography" in getattr(model, CONTRACT_ATTRIBUTE, {})

def validate_topography_tensor(x: torch.Tensor, topography: Any, context: str) -> torch.Tensor:
    if topography is None:
        raise ValueError(f"{context} requires topography data")
    if not isinstance(topography, torch.Tensor):
        raise ValueError(f"{context} topography must be a torch.Tensor")
    if topography.dtype != torch.float32:
        raise ValueError(
            f"{context} topography dtype mismatch: expected torch.float32, got {topography.dtype}"
        )
    if topography.ndim != 4:
        raise ValueError(f"{context} topography must be 4D, got {topography.ndim}D")
    expected = (int(x.shape[0]), 1, int(x.shape[-2]), int(x.shape[-1]))
    if tuple(topography.shape) != expected:
        raise ValueError(
            f"{context} topography shape mismatch: expected {expected}, got {tuple(topography.shape)}"
        )
    if not torch.isfinite(topography).all():
        raise ValueError(f"{context} topography must contain only finite values")
    return topography

def expand_topography_batch(
    x: torch.Tensor,
    topography_grid: Any,
    context: str,
) -> torch.Tensor:
    if not isinstance(topography_grid, torch.Tensor):
        raise ValueError(f"{context} static topography must be a torch.Tensor")
    expected = (1, int(x.shape[-2]), int(x.shape[-1]))
    if tuple(topography_grid.shape) != expected:
        raise ValueError(
            f"{context} static topography shape mismatch: expected {expected}, "
            f"got {tuple(topography_grid.shape)}"
        )
    batch = topography_grid.unsqueeze(0).expand(int(x.shape[0]), -1, -1, -1)
    return validate_topography_tensor(x, batch, context)
```

Change the dispatcher signature and body to:

```python
def run_uploaded_model(
    model: nn.Module,
    x: torch.Tensor,
    ls: Any = None,
    topography: Any = None,
    *,
    context: str = "uploaded model",
):
    requires_ls = uploaded_model_requires_ls(model)
    requires_topography = uploaded_model_requires_topography(model)
    if requires_ls and requires_topography:
        return model(
            x,
            validate_ls_tensor(x, ls, context),
            validate_topography_tensor(x, topography, context),
        )
    if requires_ls:
        return model(x, validate_ls_tensor(x, ls, context))
    if requires_topography:
        return model(x, validate_topography_tensor(x, topography, context))
    return model(x)
```

Do not inspect `forward`, catch `TypeError`, or retry.

- [ ] **Step 4: Verify GREEN and current Ls compatibility**

Run:

```powershell
python -m pytest AstraAtmos_backend/backend/tests/test_uploaded_model_runner.py -k "topography or auxiliary or required_ls or legacy_uploaded or ls_uploaded" -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add AstraAtmos_backend/backend/training_backbones/uploaded_model_contract.py AstraAtmos_backend/backend/tests/test_uploaded_model_runner.py
git commit -m "feat: extend uploaded model auxiliary contract"
```

### Task 2: Implement Coordinate-Aware MOLA Runtime Resampling

**Files:**
- Modify: `AstraAtmos_backend/backend/config.py`
- Create: `AstraAtmos_backend/backend/training_backbones/mola_topography.py`
- Create: `AstraAtmos_backend/backend/tests/test_mola_topography.py`

- [ ] **Step 1: Write deterministic resampling and validation tests**

Create a fixture helper that writes `elevation`, `latitude`, and `longitude`
to a temporary NetCDF. Use a periodic analytic field such as
`latitude + 10 * cos(longitude)` so expected seam behavior is explicit.

Add the fixture and representative interpolation tests directly; parameterize
the same helper for the listed validation failures:

```python
def _periodic_field(latitude, longitude):
    return (
        np.asarray(latitude, dtype=np.float64)[:, None]
        + 10.0 * np.cos(np.deg2rad(np.asarray(longitude, dtype=np.float64)))[None, :]
    ).astype(np.float32)

def _write_mola_fixture(path, latitude, longitude, elevation=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = _periodic_field(latitude, longitude) if elevation is None else elevation
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("latitude", len(latitude))
        dataset.createDimension("longitude", len(longitude))
        dataset.createVariable("latitude", "f4", ("latitude",))[:] = latitude
        dataset.createVariable("longitude", "f4", ("longitude",))[:] = longitude
        variable = dataset.createVariable("elevation", "f4", ("latitude", "longitude"))
        variable.units = "meter"
        variable[:] = values

def test_resample_preserves_descending_target_latitude():
    source_lat = np.array([-60.0, -20.0, 20.0, 60.0])
    source_lon = np.arange(-180.0, 180.0, 45.0)
    target_lat = np.array([40.0, 0.0, -40.0])
    target_lon = np.array([-135.0, -45.0, 45.0, 135.0])
    result = resample_mola(
        _periodic_field(source_lat, source_lon),
        source_lat,
        source_lon,
        target_lat,
        target_lon,
    )
    assert result.shape == (3, 4)
    assert np.all(result[0] > result[1])
    assert np.all(result[1] > result[2])

def test_resample_restores_ascending_target_latitude():
    source_lat = np.array([60.0, 20.0, -20.0, -60.0])
    source_lon = np.arange(0.0, 360.0, 45.0)
    target_lat = np.array([-40.0, 0.0, 40.0])
    result = resample_mola(
        _periodic_field(source_lat, source_lon),
        source_lat,
        source_lon,
        target_lat,
        np.array([0.0, 90.0, 180.0, 270.0]),
    )
    assert np.all(result[0] < result[1])
    assert np.all(result[1] < result[2])

def test_resample_wraps_minus_180_and_180_without_a_seam():
    source_lat = np.array([-45.0, 45.0])
    source_lon = np.arange(-180.0, 180.0, 30.0)
    target_lon = np.array([-180.1, 179.9, -180.0, 180.0, -179.9, 180.1])
    result = resample_mola(
        _periodic_field(source_lat, source_lon),
        source_lat,
        source_lon,
        np.array([0.0]),
        target_lon,
    )[0]
    assert result[0] == pytest.approx(result[1], abs=1e-5)
    assert result[2] == pytest.approx(result[3], abs=1e-5)
    assert result[4] == pytest.approx(result[5], abs=1e-5)

def test_same_shape_different_coordinates_are_interpolated_not_copied():
    source_lat = np.array([-45.0, 45.0])
    source_lon = np.array([-180.0, -90.0, 0.0, 90.0])
    source = _periodic_field(source_lat, source_lon)
    result = resample_mola(
        source,
        source_lat,
        source_lon,
        np.array([-30.0, 30.0]),
        np.array([-135.0, -45.0, 45.0, 135.0]),
    )
    assert result.shape == source.shape
    assert not np.array_equal(result, source)

@pytest.mark.parametrize(("height", "width"), [(2, 4), (4, 8), (6, 12)])
def test_resample_supports_multiple_target_grid_sizes(height, width):
    source_lat = np.linspace(-75.0, 75.0, 6)
    source_lon = np.arange(-180.0, 180.0, 30.0)
    target_lat = np.linspace(-60.0, 60.0, height)
    target_lon = np.arange(width) * (360.0 / width) - 180.0
    result = resample_mola(
        _periodic_field(source_lat, source_lon),
        source_lat,
        source_lon,
        target_lat,
        target_lon,
    )
    assert result.shape == (height, width)
    assert np.isfinite(result).all()

def test_load_rejects_missing_asset_with_path_and_target_shape(tmp_path):
    missing = tmp_path / "missing.nc"
    with pytest.raises(FileNotFoundError, match=r"missing\.nc.*target.*\(2, 4\)"):
        prepare_topography_grid(
            np.array([-45.0, 45.0]),
            np.array([-180.0, -90.0, 0.0, 90.0]),
            asset_path=missing,
        )

@pytest.mark.parametrize("missing_name", ["elevation", "latitude", "longitude"])
def test_load_rejects_missing_required_variable(tmp_path, missing_name):
    path = tmp_path / "broken.nc"
    _write_mola_fixture(path, [-45.0, 45.0], [-180.0, -90.0, 0.0, 90.0])
    with netCDF4.Dataset(path, "r+") as dataset:
        dataset.renameVariable(missing_name, f"missing_{missing_name}")
    with pytest.raises(ValueError, match=missing_name):
        load_mola_asset(path)

@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_load_rejects_non_finite_elevation(tmp_path, bad):
    path = tmp_path / "non-finite.nc"
    elevation = _periodic_field([-45.0, 45.0], [-180.0, -90.0, 0.0, 90.0])
    elevation[0, 0] = bad
    _write_mola_fixture(path, [-45.0, 45.0], [-180.0, -90.0, 0.0, 90.0], elevation)
    with pytest.raises(ValueError, match="finite"):
        load_mola_asset(path)

def test_load_rejects_incomplete_periodic_longitude_grid(tmp_path):
    path = tmp_path / "incomplete.nc"
    _write_mola_fixture(path, [-45.0, 45.0], [-180.0, -90.0, 0.0])
    with pytest.raises(ValueError, match="global period"):
        load_mola_asset(path)

def test_prepare_static_topography_returns_float32_1hw_tensor(tmp_path):
    path = tmp_path / "mola.nc"
    _write_mola_fixture(path, [-45.0, 45.0], [-180.0, -90.0, 0.0, 90.0])
    result = prepare_topography_grid(
        np.array([-30.0, 30.0]),
        np.array([-135.0, -45.0, 45.0, 135.0]),
        asset_path=path,
    )
    assert result.shape == (1, 2, 4)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()
```

For the seam test, request target longitudes
`[-180.1, -180.0, -179.9, 179.9, 180.0, 180.1]` and assert equivalent wrapped
positions agree within interpolation tolerance.

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
python -m pytest AstraAtmos_backend/backend/tests/test_mola_topography.py -q
```

Expected: collection fails because `mola_topography.py` does not exist.

- [ ] **Step 3: Add the configured asset path**

In `config.py`, add:

```python
MOLA_TOPOGRAPHY_PATH = Path(
    os.getenv(
        "ASTRAATMOS_MOLA_TOPOGRAPHY_PATH",
        DATA_DIR / "assets" / "mola_topography_5deg.nc",
    )
)
```

- [ ] **Step 4: Implement the MOLA module**

Create public APIs:

```python
def validate_rectilinear_grid(
    latitude,
    longitude,
    *,
    context: str,
    require_global_longitude: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated float64 one-dimensional target coordinates."""

def load_mola_asset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite float32 elevation plus validated source coordinates."""

def resample_mola(
    source_elevation,
    source_latitude,
    source_longitude,
    target_latitude,
    target_longitude,
) -> np.ndarray:
    """Return finite float32 elevation aligned to the target coordinate order."""

def prepare_topography_grid(
    target_latitude,
    target_longitude,
    *,
    asset_path: Path = MOLA_TOPOGRAPHY_PATH,
) -> torch.Tensor:
    """Load, resample, and return static terrain with shape [1, H, W]."""

def global_cell_center_coordinates(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    latitude = 90.0 - 90.0 / height - np.arange(height) * (180.0 / height)
    longitude = -180.0 + np.arange(width) * (360.0 / width)
    return latitude.astype(np.float32), longitude.astype(np.float32)
```

`validate_rectilinear_grid` is for source assets and actual dataset grids. It
requires finite one-dimensional strictly monotonic latitude within `[-90, 90]`,
unique longitude modulo 360, and at least two points on each axis. With
`require_global_longitude=True`, it additionally requires nearly uniform cyclic
longitude gaps and `count * spacing == 360` within tolerance.

`load_mola_asset` and uploaded training target-grid readers set this flag.
`resample_mola` uses the strict validator for its source, then separately
validates interpolation query coordinates as finite, non-empty one-dimensional
arrays with target latitudes inside the source-center range. Queries retain
their original order and may contain equivalent wrapped longitudes, which is
needed to prove -180/180 continuity; actual dataset coordinates remain strict,
unique, and globally complete before they reach the resampler.

Implement seam-free bilinear interpolation in two one-dimensional passes:

```python
lat_order = np.argsort(source_latitude)
lon_normalized = np.mod(source_longitude, 360.0)
lon_order = np.argsort(lon_normalized)
elevation = source_elevation[np.ix_(lat_order, lon_order)]
lat_sorted = source_latitude[lat_order]
lon_sorted = lon_normalized[lon_order]

lon_extended = np.concatenate(([lon_sorted[-1] - 360.0], lon_sorted, [lon_sorted[0] + 360.0]))
elevation_extended = np.concatenate(
    (elevation[:, -1:], elevation, elevation[:, :1]),
    axis=1,
)
target_lon = np.mod(np.asarray(target_longitude, dtype=np.float64), 360.0)
along_lon = np.stack(
    [np.interp(target_lon, lon_extended, row) for row in elevation_extended],
    axis=0,
)
output = np.stack(
    [np.interp(target_latitude, lat_sorted, along_lon[:, index]) for index in range(len(target_lon))],
    axis=1,
).astype(np.float32)
```

Reject target latitude outside the source-center range rather than
extrapolating. Verify the exact output shape and finite values. Include asset
path, source shape, and target shape in failures from `prepare_topography_grid`.
Return `torch.from_numpy(output).unsqueeze(0)` with shape `[1, H, W]`.

- [ ] **Step 5: Verify GREEN**

Run:

```powershell
python -m pytest AstraAtmos_backend/backend/tests/test_mola_topography.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```powershell
git add AstraAtmos_backend/backend/config.py AstraAtmos_backend/backend/training_backbones/mola_topography.py AstraAtmos_backend/backend/tests/test_mola_topography.py
git commit -m "feat: add coordinate-aware MOLA resampling"
```

### Task 3: Preserve And Validate Dataset Coordinates

**Files:**
- Modify: `AstraAtmos_backend/backend/training_backbones/user_model_runner.py`
- Modify: `AstraAtmos_backend/backend/tests/test_uploaded_model_runner.py`

- [ ] **Step 1: Write failing coordinate propagation tests**

Extend the current NetCDF helpers and add tests that request
`return_coordinates=True`:

```python
result = prepare_tensors(
    openmars_dir,
    mcd_dir,
    [],
    window=3,
    horizon=2,
    training_dataset="mcd_overview",
    mcd_overview_dir=overview_dir,
    return_ls=True,
    return_coordinates=True,
)
x, y, ls, mean, std, height, width, latitude, longitude = result
assert np.array_equal(latitude, np.array([-45.0, 45.0], dtype=np.float32))
assert np.array_equal(longitude, np.array([0.0, 120.0, 240.0], dtype=np.float32))
assert (height, width) == (len(latitude), len(longitude))
```

Add failures for a missing coordinate, coordinate length mismatch, non-finite
coordinate, non-monotonic latitude, incomplete longitude coverage, and two
dataset files with different grids. Add a raw-MCD assertion that the output
grid is latitude `87.5` through `-87.5` and longitude `-180` through `175` after the
existing 37-to-36 latitude conversion.

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
python -m pytest AstraAtmos_backend/backend/tests/test_uploaded_model_runner.py -k "coordinate or grid or raw_3h" -q
```

Expected: `return_coordinates` is unsupported and invalid grids are not
rejected.

- [ ] **Step 3: Implement coordinate readers and consistency checks**

Import `validate_rectilinear_grid` and add:

```python
def _read_spatial_coordinates(dataset: Any, file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    lat_name = "lat" if "lat" in dataset.variables else "latitude" if "latitude" in dataset.variables else None
    lon_name = "lon" if "lon" in dataset.variables else "longitude" if "longitude" in dataset.variables else None
    if lat_name is None or lon_name is None:
        raise ValueError(f"Missing latitude/longitude coordinates in {file_path}")
    return validate_rectilinear_grid(
        dataset.variables[lat_name][:],
        dataset.variables[lon_name][:],
        context=f"dataset grid in {file_path}",
        require_global_longitude=True,
    )

def _require_matching_grid(
    expected: tuple[np.ndarray, np.ndarray],
    actual: tuple[np.ndarray, np.ndarray],
    file_path: Path,
) -> None:
    if not np.array_equal(expected[0], actual[0]) or not np.array_equal(expected[1], actual[1]):
        raise ValueError(
            f"Spatial grid mismatch in {file_path}: expected "
            f"({len(expected[0])}, {len(expected[1])}), got "
            f"({len(actual[0])}, {len(actual[1])})"
        )
```

Return coordinates from `_load_openmars`, `_load_mcd_overview`,
`_load_raw_3h_mcd`, and `_load_mcd_full_training_dataset`. For raw MCD, use:

```python
target_latitude = RAW_MCD_TARGET_LAT.copy()
target_longitude = np.asarray(lon_values, dtype=np.float32)[:72]
```

Validate every contributing target file against the first grid. Pass the
OpenMars target grid into `_load_mcd_features` and validate every selected MCD
feature file against it before accepting equal-sized arrays.

Add `return_coordinates: bool = False` to `prepare_tensors`. Preserve both
existing return signatures exactly. When true, append `latitude, longitude`
after `height, width`; require coordinate lengths to match the post-loading
`height, width` instead of silently clipping the grid.

- [ ] **Step 4: Verify GREEN and old tuple compatibility**

Run:

```powershell
python -m pytest AstraAtmos_backend/backend/tests/test_uploaded_model_runner.py -k "prepare_tensors or coordinate or grid or raw_3h" -q
```

Expected: new coordinate tests and all existing six-/seven-item return tests
pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add AstraAtmos_backend/backend/training_backbones/user_model_runner.py AstraAtmos_backend/backend/tests/test_uploaded_model_runner.py
git commit -m "feat: preserve uploaded training spatial grids"
```

### Task 4: Build And Commit The Verified 36 x 72 MOLA Asset

**Files:**
- Create: `AstraAtmos_backend/backend/scripts/build_mola_topography_asset.py`
- Create: `AstraAtmos_backend/backend/data/assets/mola_topography_5deg.nc`
- Create: `docs/mola-topography-asset.md`
- Modify: `AstraAtmos_backend/backend/tests/test_mola_topography.py`

- [ ] **Step 1: Write a failing committed-asset integrity test**

Add a test against `config.MOLA_TOPOGRAPHY_PATH` asserting:

```python
with netCDF4.Dataset(config.MOLA_TOPOGRAPHY_PATH) as dataset:
    assert set(("elevation", "latitude", "longitude")) <= set(dataset.variables)
    assert dataset.variables["elevation"].dtype == np.dtype("float32")
    assert dataset.variables["elevation"].shape == (36, 72)
    assert dataset.variables["elevation"].units == "meter"
    assert np.array_equal(dataset.variables["latitude"][:], np.arange(87.5, -90.0, -5.0, dtype=np.float32))
    assert np.array_equal(dataset.variables["longitude"][:], np.arange(-180.0, 180.0, 5.0, dtype=np.float32))
    assert dataset.source_product_id == "MEGT90N000CB.IMG"
    assert dataset.source_sha256 == "25f16fb7aaf857898dcf98bc4f841341a24f8b9f7e98453ca083bc45d897ca2c"
    assert np.isfinite(dataset.variables["elevation"][:]).all()
```

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
python -m pytest AstraAtmos_backend/backend/tests/test_mola_topography.py -k committed -q
```

Expected: failure because the committed asset does not exist.

- [ ] **Step 3: Implement the reproducible builder**

The script constants are:

```python
SOURCE_BASE = "https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg004"
IMAGE_NAME = "megt90n000cb.img"
LABEL_NAME = "megt90n000cb.lbl"
IMAGE_BYTES = 2_073_600
IMAGE_SHA256 = "25f16fb7aaf857898dcf98bc4f841341a24f8b9f7e98453ca083bc45d897ca2c"
SOURCE_SHAPE = (720, 1440)
TARGET_LATITUDE = np.arange(87.5, -90.0, -5.0, dtype=np.float32)
TARGET_LONGITUDE = np.arange(-180.0, 180.0, 5.0, dtype=np.float32)
```

Provide CLI arguments `--output`, optional `--source-dir`, and `--validate`.
Download with `urllib.request.urlopen` only when the exact source file is not
present. Verify label fields `LINES=720`, `LINE_SAMPLES=1440`,
`SAMPLE_TYPE=MSB_INTEGER`, `SAMPLE_BITS=16`, `UNIT=METER`, and
`MAP_RESOLUTION=4.0`; verify image byte count and SHA-256 before decoding with:

```python
source = np.fromfile(image_path, dtype=">i2").reshape(SOURCE_SHAPE).astype(np.float32)
source_latitude = 89.875 - np.arange(720, dtype=np.float64) * 0.25
source_longitude = 0.125 + np.arange(1440, dtype=np.float64) * 0.25
```

For each target center, average all 0.25-degree cells within its 5-degree
latitude band and within periodic angular longitude distance `< 2.5` degrees.
Each output cell must aggregate exactly 400 source cells. Write compressed
NetCDF variables with `zlib=True`, `complevel=4`, and global provenance
attributes from the approved design. `--validate` calls the same runtime asset
loader, checks exact shape/coordinates/dtype/unit/provenance, and exits nonzero
on any mismatch.

- [ ] **Step 4: Generate and validate the platform asset**

Run:

```powershell
python AstraAtmos_backend/backend/scripts/build_mola_topography_asset.py --output AstraAtmos_backend/backend/data/assets/mola_topography_5deg.nc
python AstraAtmos_backend/backend/scripts/build_mola_topography_asset.py --output AstraAtmos_backend/backend/data/assets/mola_topography_5deg.nc --validate
```

Expected: the first command downloads/verifies the 2,073,600-byte PDS image
and writes a finite 36 x 72 asset; the second reports successful validation.
The raw `.IMG` and `.LBL` stay outside the repository.

- [ ] **Step 5: Document provenance and preprocessing**

Create `docs/mola-topography-asset.md` containing the archive ID,
`MEGT90N000CB.IMG`, both PDS source URLs, 0.25-degree original resolution,
source SHA-256 above, 5-degree periodic area aggregation rule, output variable
schema, and exact reproduction/validation commands from Step 4.

- [ ] **Step 6: Verify GREEN**

Run:

```powershell
python -m pytest AstraAtmos_backend/backend/tests/test_mola_topography.py -q
```

Expected: all unit and committed-asset tests pass.

- [ ] **Step 7: Commit Task 4**

```powershell
git add AstraAtmos_backend/backend/scripts/build_mola_topography_asset.py AstraAtmos_backend/backend/data/assets/mola_topography_5deg.nc AstraAtmos_backend/backend/tests/test_mola_topography.py docs/mola-topography-asset.md
git commit -m "feat: add verified MOLA topography asset"
```

### Task 5: Use Real MOLA During Uploaded-Model Dry-Run

**Files:**
- Modify: `AstraAtmos_backend/backend/services/user_model_validator.py`
- Modify: `AstraAtmos_backend/backend/tests/test_user_model_validator.py`

- [ ] **Step 1: Write failing topography validator tests**

Add topography-only and combined uploaded source strings. Their `forward`
methods assert `torch.float32`, expected `[2, 1, 8, 16]`, finite values, and
non-flat terrain. The combined source records argument order by independently
using `ls.shape` and `topography.shape`.

Add tests using the complete topography sources and helper already defined at
the top of this test module:

```python
def test_topography_model_dry_run_receives_real_mola_tensor():
    result = _validate_source(TOPOGRAPHY_MODEL_SOURCE)
    assert result.ok is True
    assert result.errors == []
    assert result.output_shape == [2, 3, 1, 8, 16]

def test_ls_topography_model_dry_run_receives_fixed_argument_order():
    result = _validate_source(LS_TOPOGRAPHY_MODEL_SOURCE)
    assert result.ok is True
    assert result.errors == []

def test_topography_model_dry_run_fails_when_mola_asset_is_missing(monkeypatch, tmp_path):
    missing = tmp_path / "missing-mola.nc"
    monkeypatch.setattr(user_model_validator, "MOLA_TOPOGRAPHY_PATH", missing)
    result = _validate_source_with_validator(
        TOPOGRAPHY_MODEL_SOURCE,
        UserModelValidator(timeout_seconds=None),
    )
    assert result.ok is False
    assert any(str(missing) in error and "(8, 16)" in error for error in result.errors)

def test_legacy_and_ls_dry_runs_do_not_load_mola(monkeypatch):
    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("MOLA must not load for undeclared topography")
    monkeypatch.setattr(user_model_validator, "prepare_topography_grid", fail_if_loaded)
    assert _validate_source_with_validator(
        VALID_MODEL_SOURCE,
        UserModelValidator(timeout_seconds=None),
    ).ok
    assert _validate_source_with_validator(
        LS_MODEL_SOURCE,
        UserModelValidator(timeout_seconds=None),
    ).ok
```

For the missing-asset test, monkeypatch the validator module's configured path
to `tmp_path / "missing-mola.nc"` and assert the result names that path and the
dry-run target shape `(8, 16)`.

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
python -m pytest AstraAtmos_backend/backend/tests/test_user_model_validator.py -k "topography or legacy_and_ls" -q
```

Expected: topography models fail because dry-run dispatch has no terrain.

- [ ] **Step 3: Prepare real terrain only for declared models**

Import `MOLA_TOPOGRAPHY_PATH`, `global_cell_center_coordinates`,
`prepare_topography_grid`, `expand_topography_batch`, and
`uploaded_model_requires_topography`. After attaching the contract:

```python
x = torch.zeros(2, 3, 1, 8, 16)
ls = torch.zeros(2, 3, dtype=torch.float32)
topography_batch = None
if uploaded_model_requires_topography(model):
    target_latitude, target_longitude = global_cell_center_coordinates(8, 16)
    static_topography = prepare_topography_grid(
        target_latitude,
        target_longitude,
        asset_path=MOLA_TOPOGRAPHY_PATH,
    )
    topography_batch = expand_topography_batch(
        x,
        static_topography,
        "uploaded model dry-run",
    )
output = run_uploaded_model(
    model,
    x,
    ls=ls,
    topography=topography_batch,
    context="uploaded model dry-run",
)
```

Do not call MOLA helpers when topography is undeclared. Preserve existing error
wrapping so missing/corrupt asset details appear after `Model dry-run failed:`.

- [ ] **Step 4: Verify GREEN and complete validator compatibility**

Run:

```powershell
python -m pytest AstraAtmos_backend/backend/tests/test_user_model_validator.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 5**

```powershell
git add AstraAtmos_backend/backend/services/user_model_validator.py AstraAtmos_backend/backend/tests/test_user_model_validator.py
git commit -m "feat: validate uploaded models with real MOLA terrain"
```

### Task 6: Reuse Static Topography During Training, Validation, And Metrics

**Files:**
- Modify: `AstraAtmos_backend/backend/training_backbones/user_model_runner.py`
- Modify: `AstraAtmos_backend/backend/tests/test_uploaded_model_runner.py`

- [ ] **Step 1: Write failing runner propagation tests**

Add `TOPOGRAPHY_RUNNER_MODEL_SOURCE` whose model records/asserts every terrain
shape and dtype while `window=3`, `horizon=2`. Add tests that:

- `_forward_uploaded_batch` expands the same static `[1, H, W]` storage for
  batch sizes 2 and 1;
- a topography-only model trains when its data has no Ls;
- a combined model receives both inputs in train, validation, and final metric
  passes;
- missing and corrupt assets fail before the first optimizer step;
- `window != horizon` leaves each terrain call at `[B, 1, H, W]`.

Monkeypatch `prepare_topography_grid` in loop-unit tests to return a known
non-flat tensor and record that it is called exactly once per `main()` run.

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
python -m pytest AstraAtmos_backend/backend/tests/test_uploaded_model_runner.py -k "topography and (forward or main or training or window)" -q
```

Expected: training never prepares or passes topography.

- [ ] **Step 3: Add static topography to the batch helper**

Change `_forward_uploaded_batch` to accept
`topography_grid: Optional[torch.Tensor] = None`. After moving `x`, `y`, and Ls
to the device:

```python
topography_batch = None
if uploaded_model_requires_topography(model):
    topography_batch = expand_topography_batch(
        x_device,
        topography_grid,
        context,
    )
prediction = run_uploaded_model(
    model,
    x_device,
    ls=ls_device,
    topography=topography_batch,
    context=context,
)
```

Keep DataLoader datasets as `(x, y)` or `(x, ls, y)` only; never repeat terrain
per sample or along `window`.

- [ ] **Step 4: Prepare terrain once before training**

Call `prepare_tensors` with `return_ls=True, return_coordinates=True` and unpack
the nine values. After loading the model:

```python
model_topography = None
if uploaded_model_requires_topography(model):
    model_topography = prepare_topography_grid(
        target_latitude,
        target_longitude,
        asset_path=MOLA_TOPOGRAPHY_PATH,
    ).to(device)
    expand_topography_batch(
        x_torch[:1].to(device),
        model_topography,
        "uploaded training dataset",
    )
```

Pass `model_topography` to `_forward_uploaded_batch` in the training loop,
validation loop, and final metric loop. Preserve existing Ls preflight and
ensure topography-only models do not require Ls. Do not save terrain in the
model state dict or checkpoint.

- [ ] **Step 5: Verify GREEN and old runner behavior**

Run:

```powershell
python -m pytest AstraAtmos_backend/backend/tests/test_uploaded_model_runner.py -q
```

Expected: all runner tests pass.

- [ ] **Step 6: Commit Task 6**

```powershell
git add AstraAtmos_backend/backend/training_backbones/user_model_runner.py AstraAtmos_backend/backend/tests/test_uploaded_model_runner.py
git commit -m "feat: pass static MOLA terrain through uploaded training"
```

### Task 7: Propagate Topography Through All Uploaded Inference Paths

**Files:**
- Modify: `AstraAtmos_backend/backend/services/inference_service.py`
- Modify: `AstraAtmos_backend/backend/tests/test_uploaded_model_ls_inference.py`

- [ ] **Step 1: Write failing inference path tests**

Generalize the existing setup to a recording combined model and make fake
`prepare_tensors` require both `return_ls=True` and
`return_coordinates=True`. Return known latitude/longitude after the existing
seven values. Monkeypatch `prepare_topography_grid` to return a non-flat static
tensor.

Add or extend concrete tests following the existing setup. Each assertion uses
the recording model returned by `_setup_uploaded_inference`:

```python
def test_uploaded_formal_prediction_receives_matching_ls_and_topography(monkeypatch):
    service, task, hypers, model, ls, static_topography = _setup_uploaded_inference(monkeypatch)
    service._predict_uploaded_task_window(task, hypers, ls_start=0.0, horizon=2)
    assert torch.equal(model.ls_calls[-1], ls[0:1])
    assert torch.equal(model.topography_calls[-1][0], static_topography)
    assert model.topography_calls[-1].shape == (1, 1, 2, 2)

def test_uploaded_test_set_metrics_receives_topography_for_each_batch(monkeypatch):
    service, task, hypers, model, ls, static_topography = _setup_uploaded_inference(monkeypatch)
    service._uploaded_task_test_set_arrays(task, hypers, horizon=2)
    assert torch.equal(model.ls_calls[-1], ls[8:10])
    assert model.topography_calls[-1].shape == (2, 1, 2, 2)
    assert torch.equal(model.topography_calls[-1][0], static_topography)
    assert torch.equal(model.topography_calls[-1][1], static_topography)

def test_uploaded_test_results_receive_topography(monkeypatch):
    service, task, hypers, model, _ls, static_topography = _setup_uploaded_inference(monkeypatch)
    asyncio.run(service._get_uploaded_model_test_results(task, hypers))
    assert torch.equal(model.topography_calls[-1][0], static_topography)

def test_uploaded_permutation_importance_keeps_ls_and_topography_fixed(monkeypatch):
    service, task, hypers, model, ls, static_topography = _setup_uploaded_inference(monkeypatch)
    service._uploaded_task_permutation_importance(
        task,
        hypers,
        selected_variables=["Ozone"],
        horizon=2,
    )
    assert len(model.topography_calls) >= 2
    assert all(torch.equal(call[0], static_topography) for call in model.topography_calls)
    assert all(torch.equal(call, ls[8:10]) for call in model.ls_calls)

def test_legacy_and_ls_only_inference_do_not_prepare_mola(monkeypatch):
    service, task, hypers, _model, _ls, _static = _setup_uploaded_inference(
        monkeypatch,
        model_kind="ls",
    )
    monkeypatch.setattr(
        inference_service,
        "prepare_topography_grid",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("MOLA must not load for an Ls-only model")
        ),
    )
    service._predict_uploaded_task_window(task, hypers, ls_start=0.0, horizon=2)
```

Record each received terrain tensor and assert every permutation call is equal
to the baseline terrain while only `x` changes.

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
python -m pytest AstraAtmos_backend/backend/tests/test_uploaded_model_ls_inference.py -q
```

Expected: tuple unpacking or model calls fail because coordinates/topography
are not propagated.

- [ ] **Step 3: Centralize uploaded inference preparation**

Change `_prepare_uploaded_task_tensors` to request coordinates and return the
same selected channels plus nine-value tensor tuple. Add a focused helper:

```python
def _prepare_uploaded_topography(self, model, latitude, longitude):
    if not uploaded_model_requires_topography(model):
        return None
    return prepare_topography_grid(
        latitude,
        longitude,
        asset_path=MOLA_TOPOGRAPHY_PATH,
    ).to(self.device)

def _run_uploaded_task_model(self, model, x_batch, ls_batch, topography_grid, context):
    x_device = x_batch.to(self.device)
    topography_batch = None
    if uploaded_model_requires_topography(model):
        topography_batch = expand_topography_batch(x_device, topography_grid, context)
    return run_uploaded_model(
        model,
        x_device,
        ls=ls_batch.to(self.device) if ls_batch is not None else None,
        topography=topography_batch,
        context=context,
    )
```

This helper is the only uploaded-model call in the four inference methods.

- [ ] **Step 4: Replace each inference call**

In formal prediction, test-set arrays/metrics, permutation importance, and
uploaded test results:

1. unpack `target_latitude, target_longitude`;
2. prepare one static grid after loading the model;
3. call `_run_uploaded_task_model` for each selected batch;
4. retain existing output-shape checks and metric conversion.

Permutation `score(batch)` closes over unchanged `ls_sample` and
`topography_grid`; it only receives the baseline or feature-shuffled `x`.

- [ ] **Step 5: Verify GREEN and prediction regressions**

Run:

```powershell
python -m pytest AstraAtmos_backend/backend/tests/test_uploaded_model_ls_inference.py AstraAtmos_backend/backend/tests/test_trained_model_predict_contract.py AstraAtmos_backend/backend/tests/test_prediction_horizon_contract.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 7**

```powershell
git add AstraAtmos_backend/backend/services/inference_service.py AstraAtmos_backend/backend/tests/test_uploaded_model_ls_inference.py
git commit -m "feat: pass MOLA terrain through uploaded inference"
```

### Task 8: Update The Upload Template And Guide Without Changing The Default AST

**Files:**
- Modify: `docs/uploaded-model-template.py`
- Modify: `docs/uploaded-model-training.md`
- Modify: `AstraAtmos_backend/backend/tests/test_user_model_downloads.py`

- [ ] **Step 1: Write failing template and guide tests**

Parse the template and assert the executable class still has exactly:

```python
module = ast.parse(template)
forward = next(
    node
    for node in ast.walk(module)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward"
)
assert [argument.arg for argument in forward.args.args] == ["self", "x"]
```

Also assert the downloaded text contains canonical topography metadata,
`forward(self, x, topography)`, `forward(self, x, ls, topography)`,
`[batch, 1, height, width]`, `float32`, `meter`, and a terrain-fusion example.

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
python -m pytest AstraAtmos_backend/backend/tests/test_user_model_downloads.py -q
```

Expected: new content assertions fail.

- [ ] **Step 3: Add comment-only opt-in examples**

Keep the live `MODEL_SPEC`, class, `forward(self, x)`, and `build_model(config)`
unchanged. Add commented canonical topography-only and combined metadata plus
commented methods showing:

```python
# def forward(self, x, topography):
#     terrain_features = self.terrain_encoder(topography)
#     temporal_features = self.encoder(x[:, -1])
#     fused = temporal_features + terrain_features
#     return self.head(fused).unsqueeze(1).repeat(1, self.horizon, 1, 1, 1)

# def forward(self, x, ls, topography):
#     terrain_features = self.terrain_encoder(topography)
#     phase_features = self.phase_encoder(ls)
#     phase_bias = self.phase_encoder(ls).unsqueeze(-1).unsqueeze(-1)
#     fused = temporal_features + terrain_features + phase_bias
#     return self.head(fused).unsqueeze(1).repeat(1, self.horizon, 1, 1, 1)
```

Explain that topography is static, meter-valued `float32`, has no time/window
axis, and is never included in `in_channels`.

- [ ] **Step 4: Extend the public guide**

Document all four declarations and calls, strict metadata, coordinate-based
MOLA alignment, errors, real-MOLA dry-run, batch expansion, permutation
behavior, and backward compatibility. Link `mola-topography-asset.md` for
provenance and reproduction details.

- [ ] **Step 5: Verify GREEN**

Run:

```powershell
python -m pytest AstraAtmos_backend/backend/tests/test_user_model_downloads.py -q
```

Expected: all tests pass, including the executable AST assertion.

- [ ] **Step 6: Commit Task 8**

```powershell
git add docs/uploaded-model-template.py docs/uploaded-model-training.md AstraAtmos_backend/backend/tests/test_user_model_downloads.py
git commit -m "docs: document uploaded model topography input"
```

### Task 9: Full Verification And Scope Audit

**Files:**
- Verify all files above; make no unrelated edits.

- [ ] **Step 1: Validate the committed data asset independently**

Run:

```powershell
python AstraAtmos_backend/backend/scripts/build_mola_topography_asset.py --output AstraAtmos_backend/backend/data/assets/mola_topography_5deg.nc --validate
```

Expected: exact 36 x 72 shape, canonical coordinates, `float32` meter values,
finite data, complete provenance, and source checksum all validate.

- [ ] **Step 2: Run focused topography and uploaded-model suites**

Run:

```powershell
python -m pytest AstraAtmos_backend/backend/tests/test_mola_topography.py AstraAtmos_backend/backend/tests/test_uploaded_model_runner.py AstraAtmos_backend/backend/tests/test_user_model_validator.py AstraAtmos_backend/backend/tests/test_uploaded_model_ls_inference.py AstraAtmos_backend/backend/tests/test_user_model_downloads.py -q
```

Expected: zero failures and no new warnings.

- [ ] **Step 3: Run existing training and prediction regressions**

Run:

```powershell
python -m pytest AstraAtmos_backend/backend/tests/test_uploaded_training_contract.py AstraAtmos_backend/backend/tests/test_training_dataset_loader.py AstraAtmos_backend/backend/tests/test_training_model_artifacts.py AstraAtmos_backend/backend/tests/test_trained_model_test_set_metrics.py AstraAtmos_backend/backend/tests/test_prediction_horizon_contract.py AstraAtmos_backend/backend/tests/test_trained_model_predict_contract.py -q
```

Expected: zero failures.

- [ ] **Step 4: Run the complete backend test suite**

Run:

```powershell
python -m pytest AstraAtmos_backend/backend/tests -q
```

Expected: zero failures. Record any environment-only skip or failure with its
exact test name and error rather than claiming the suite passed.

- [ ] **Step 5: Audit prohibited implementation patterns**

Run:

```powershell
rg -n "inspect\.signature|except TypeError|topography.*repeat|cat\(.*topography|zeros.*topography|rand.*topography" AstraAtmos_backend/backend
```

Expected: no uploaded-model dispatch fallback, time repetition, feature-channel
insertion, or synthetic terrain fallback. Review any legitimate test matches
manually.

- [ ] **Step 6: Check repository scope and formatting**

Run:

```powershell
git diff --check
git status --short
git log --oneline -10
```

Expected: no whitespace errors, only planned project files changed, the raw
PDS `.IMG/.LBL` are absent, and the generated NetCDF asset is tracked.
