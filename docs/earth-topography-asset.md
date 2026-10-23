# Earth ETOPO atmospheric topography asset

The Earth topography input is an offline, content-identified NetCDF asset. It
is independent of Mars MOLA and is loaded only when an uploaded Earth model
declares `MODEL_SPEC.derived_inputs.topography`.

## Source and semantics

The reproducible source is NOAA NCEI **ETOPO 2022 v1, 60 arc-second global
surface** (`ETOPO_2022_v1_60s_N90W180_surface.nc`). The official product is a
cell-registered global grid in WGS84 geographic coordinates (EPSG:4326), with
float32 elevations in metres relative to the EGM2008 geoid (EPSG:3855). The
official ETOPO guide states that the 30/60-second grids are downsampled from
the 15-second tiles and that NoData is `-99999` (not present in this source).
The Ice Surface variant represents the upper surface of Greenland and
Antarctic ice sheets.

The builder accepts only the reviewed byte identity for this source: SHA-256
`74bbc5c23e85188d93f0beba8116babf69e1a8f160e8dc9f053bf8df01b197b6` and
478,290,125 bytes. The upstream NetCDF does not carry a reliable surface/bed
variant field, so this is a project review pin confirmed against the NOAA
THREDDS catalogue and source URL, not a checksum published by NOAA. Renaming a
file or supplying a caller-selected digest cannot bypass the pin. Other ETOPO
resolutions, versions, bedrock files, or repackaged bytes require independent
review before use.

The platform uses the independent NOAA/NCEI GSHHG 2.3.7 intermediate coastline
binary (`gshhs_i.b`, SHA-256 recorded in the asset identity). The source pixel
mask uses the GSHHG land hierarchy and Antarctic ice-front; ocean and inland
water levels are explicitly set to zero before aggregation. This is never
inferred from `elevation < 0`, so negative-elevation land remains valid.

## Asset and identity

`earth_etopo2022_surface_0p75deg.nc` contains `elevation(latitude,longitude)`
float32 metres and the exact Earth three-hour release coordinates and
`lat_bounds`/`lon_bounds` (`240 x 480`, 0.75 degrees). Target coordinates are
read from the actual release, never generated from array dimensions. Each
target cell is a spherical-area-weighted overlap mean of source pixels after
the source land/water policy. Source rows are read in bounded windows.

The asset stores provenance excluding its own digest fields. Runtime recomputes
the semantic content digest, target-grid digest, and complete NetCDF file
SHA-256 on every load. The identity includes the source file inventory and
hashes, GSHHG mask identity, product variant/resolution, datums, policies and
regrid algorithm. Unknown or missing fields, changed bytes (including a same
size/mtime replacement), grid shifts, NoData, or non-finite values fail closed.

## Build and validate

Raw source files belong outside the checkout. The current reproducibility cache
used for the verified build is `<workspace>\earth-terrain-cache`（`<workspace>` 替换为仓库父级工作区目录） (not a
runtime dependency). From `AstraAtmos_backend/backend`:

```powershell
& "$env:CONDA_PREFIX\python.exe" scripts\build_earth_topography_asset.py `
  --source <workspace>\earth-terrain-cache\ETOPO_2022_v1_60s_N90W180_surface.nc `
  --land-mask <workspace>\earth-terrain-cache\gshhs_i_2.3.7.b `
  --resolution 60 `
  --source-sha256 74bbc5c23e85188d93f0beba8116babf69e1a8f160e8dc9f053bf8df01b197b6 `
  --land-mask-sha256 7d44bf4e16efe6056ff9d147ac0af189bbf0c16a31608426ef63ff8a91c418db `
  --output data\assets\earth_etopo2022_surface_0p75deg.nc
& "$env:CONDA_PREFIX\python.exe" scripts\build_earth_topography_asset.py `
  --validate-only --output data\assets\earth_etopo2022_surface_0p75deg.nc
```

`--download` is explicit and only downloads to `--cache-dir`; the application
never downloads source data. Build publication uses a lock, a temporary
NetCDF, full source/mask rehashes and atomic replacement. Existing valid output
is retained if validation or publication fails. A stale `*.build.lock` names
one exact file and must be investigated before removal.

The runtime API is in
`AstraAtmos_backend/backend/services/earth_topography.py`:

* `load_earth_topography(...)` returns `EarthTopographyAsset` with read-only
  arrays, `identity`, coordinates and bounds. Omitting coordinates resolves
  the actual configured three-hour release grid.
* `validate_identity(identity)` validates the frozen schema.
* `prepare_topography_batch(asset, inputs, spatial_slices)` creates independent
  `[B,1,H,W]` float32 tensors on `inputs.device`. v1 supplies one explicit
  latitude/longitude slice per sample; v2 omits slices for a full-grid copy.

The same NetCDF process lock used by MOLA is held for every source or asset
read. The path is configured by `ASTRAATMOS_EARTH_TOPOGRAPHY_PATH`; missing
assets do not affect models that do not declare terrain.

## Use and citation

The NOAA ETOPO 2022 guide states that the tiles are available for private,
academic, and commercial use. Cite: NOAA National Centers for Environmental
Information. 2022. *ETOPO 2022 15 Arc-Second Global Relief Model*.
https://doi.org/10.25921/fd45-gt74 (access date should be supplied by the
consumer). This project uses the reviewed 60 arc-second Ice Surface product.

The independent GSHHG 2.3.7 coastline mask is distributed under LGPL version 3
or later, with its copyright and permission notice retained in the source
cache. GSHHG attribution and its disclaimer must remain in supporting
documentation when the asset or build tooling is redistributed.

## Evidence

The verified build used 933,120,000 bytes of source float32 values, 90 source
rows per window (7,776,000 bytes) and a 128 MiB NetCDF chunk cache. It produced
the `[240, 480]` asset with finite range `-289.3035` to `5468.607` metres.
The source and mask content digests were respectively
`74bbc5c23e85188d93f0beba8116babf69e1a8f160e8dc9f053bf8df01b197b6` and
`7d44bf4e16efe6056ff9d147ac0af189bbf0c16a31608426ef63ff8a91c418db`.
Coarse target-cell checks retain plausible regional means (Tibetan Plateau
about 5,024 m, Greenland about 3,062 m, Antarctic interior about 2,576 m) and
set tested Pacific/Atlantic ocean cells to zero; these are cell means, not
mountain peak assertions.
`tests/test_earth_topography.py` covers spherical alignment, date-line and
polar coverage, land/ocean semantics, NoData and contract failures, identity
tampering, v1/v2 batch isolation, gradient use, atomic publication and lock
sharing. It also verifies the reviewed ETOPO content pin and rejects renamed,
repacked, bedrock, wrong-version, and wrong-resolution source claims. The
checkout-external `scripts/smoke_earth_topography.py` performs a bounded
three-day real MERRA-2 + real ETOPO v1/v2 upload, one-epoch train, strict
reload, historical forecast, and diagnostic smoke; it is not an accuracy
acceptance run.

This is a static model input asset; it does not add a three-dimensional Earth
renderer or claim predictive accuracy improvement.

## Independent 5 degree training asset

The 36×72 training release uses a separate `earth_etopo2022_surface_5deg.nc` asset and `ASTRAATMOS_EARTH_TOPOGRAPHY_5DEG_PATH`. Build from the same reviewed ETOPO/GSHHG inputs with `--target-release` pointing to the actual 5° release and a separate `--output`; runtime selects it only for that release/grid. Source policies, content and grid identity checks remain identical. See [the 5° dataset](earth-merra2-5deg.md).
