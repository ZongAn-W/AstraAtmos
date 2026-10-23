# MOLA Topography Platform Asset

## Source

- Archive: NASA/PDS `MGS-M-MOLA-5-MEGDR-L3-V1`
- Product: `MEGT90N000CB.IMG`, global median topography
- Instrument: Mars Orbiter Laser Altimeter (MOLA), Mars Global Surveyor
- Original grid: 720 x 1440, 4 pixels/degree (0.25 degree)
- Elevation representation: signed 16-bit big-endian integer, meter
- Image: `https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg004/megt90n000cb.img`
- Label: `https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg004/megt90n000cb.lbl`
- Image SHA-256: `25f16fb7aaf857898dcf98bc4f841341a24f8b9f7e98453ca083bc45d897ca2c`

The PDS product defines topography as planetary radius minus the GMM3 areoid
radius at each longitude and latitude.

## Platform Processing

The builder validates the PDS label, exact 2,073,600-byte image length, and
SHA-256 before decoding. Each output cell is the arithmetic mean of exactly
400 source cells in its periodic 5-degree latitude/longitude area. Longitude
distance is computed cyclically, so the cell centered at -180 degrees combines
source values from both sides of the 0/360 degree boundary without a seam.

The committed compressed NetCDF contains:

- `elevation(latitude, longitude)`: `float32`, unit `meter`, shape `[36, 72]`
- `latitude`: `float32`, 87.5 through -87.5 degrees
- `longitude`: `float32`, -180 through 175 degrees
- source archive, product, URLs, resolution, checksum, and preprocessing
  attributes

Runtime alignment still uses coordinates rather than array dimensions. A valid
future rectilinear target grid may receive a different height and width, with
physical detail limited by this 5-degree platform asset.

## Reproduction

From the repository root:

```powershell
python AstraAtmos_backend/backend/scripts/build_mola_topography_asset.py --output AstraAtmos_backend/backend/data/assets/mola_topography_5deg.nc
python AstraAtmos_backend/backend/scripts/build_mola_topography_asset.py --output AstraAtmos_backend/backend/data/assets/mola_topography_5deg.nc --validate
```

The raw PDS image and label are cached in the operating system temporary
directory and are not committed to the repository.
