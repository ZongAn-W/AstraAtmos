from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import sys
import tempfile
from urllib.request import urlopen

import netCDF4
import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training_backbones.mola_topography import load_mola_asset  # noqa: E402


SOURCE_BASE = (
    "https://pds-geosciences.wustl.edu/mgs/"
    "mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg004"
)
IMAGE_NAME = "megt90n000cb.img"
LABEL_NAME = "megt90n000cb.lbl"
IMAGE_BYTES = 2_073_600
IMAGE_SHA256 = (
    "25f16fb7aaf857898dcf98bc4f841341a24f8b9f7e98453ca083bc45d897ca2c"
)
SOURCE_SHAPE = (720, 1440)
TARGET_LATITUDE = np.arange(87.5, -90.0, -5.0, dtype=np.float32)
TARGET_LONGITUDE = np.arange(-180.0, 180.0, 5.0, dtype=np.float32)
REQUIRED_PROVENANCE = {
    "source_archive": "MGS-M-MOLA-5-MEGDR-L3-V1",
    "source_product_id": "MEGT90N000CB.IMG",
    "source_product_resolution": "4 pixels per degree (0.25 degree)",
    "source_image_url": f"{SOURCE_BASE}/{IMAGE_NAME}",
    "source_label_url": f"{SOURCE_BASE}/{LABEL_NAME}",
    "source_sha256": IMAGE_SHA256,
    "preprocessing": (
        "Periodic 5-degree cell aggregation; arithmetic mean of exactly "
        "400 source 0.25-degree MOLA median-topography cells per output cell"
    ),
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as response, destination.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)


def _ensure_sources(source_dir: Path) -> tuple[Path, Path]:
    image_path = source_dir / IMAGE_NAME
    label_path = source_dir / LABEL_NAME
    if not image_path.is_file():
        _download(f"{SOURCE_BASE}/{IMAGE_NAME}", image_path)
    if not label_path.is_file():
        _download(f"{SOURCE_BASE}/{LABEL_NAME}", label_path)
    return image_path, label_path


def _label_value(label: str, field: str) -> str:
    match = re.search(rf"(?m)^\s*{re.escape(field)}\s*=\s*([^\r\n]+)", label)
    if match is None:
        raise ValueError(f"PDS label is missing field {field}")
    return match.group(1).strip().strip('"')


def _validate_source(image_path: Path, label_path: Path) -> str:
    label = label_path.read_text(encoding="ascii")
    expected_fields = {
        "LINES": "720",
        "LINE_SAMPLES": "1440",
        "SAMPLE_TYPE": "MSB_INTEGER",
        "SAMPLE_BITS": "16",
        "UNIT": "METER",
    }
    for field, expected in expected_fields.items():
        actual = _label_value(label, field)
        if actual != expected:
            raise ValueError(
                f"PDS label {field} expected {expected!r}, got {actual!r}"
            )
    resolution = _label_value(label, "MAP_RESOLUTION")
    if not resolution.startswith("4.0"):
        raise ValueError(
            f"PDS label MAP_RESOLUTION expected 4.0 PIXEL/DEGREE, got {resolution!r}"
        )
    byte_count = image_path.stat().st_size
    if byte_count != IMAGE_BYTES:
        raise ValueError(
            f"PDS image byte count expected {IMAGE_BYTES}, got {byte_count}: "
            f"{image_path}"
        )
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if digest != IMAGE_SHA256:
        raise ValueError(
            f"PDS image SHA-256 expected {IMAGE_SHA256}, got {digest}: {image_path}"
        )
    return label


def _aggregate_5_degree_grid(source: np.ndarray) -> np.ndarray:
    source_latitude = 89.875 - np.arange(SOURCE_SHAPE[0], dtype=np.float64) * 0.25
    source_longitude = 0.125 + np.arange(SOURCE_SHAPE[1], dtype=np.float64) * 0.25
    output = np.empty((len(TARGET_LATITUDE), len(TARGET_LONGITUDE)), dtype=np.float32)

    for lat_index, target_latitude in enumerate(TARGET_LATITUDE.astype(np.float64)):
        lat_mask = np.abs(source_latitude - target_latitude) < 2.5
        if np.count_nonzero(lat_mask) != 20:
            raise ValueError(
                f"Latitude bin {target_latitude} expected 20 source rows, "
                f"got {np.count_nonzero(lat_mask)}"
            )
        latitude_values = source[lat_mask]
        for lon_index, target_longitude in enumerate(TARGET_LONGITUDE.astype(np.float64)):
            angular_distance = np.abs(
                (source_longitude - target_longitude + 180.0) % 360.0 - 180.0
            )
            lon_mask = angular_distance < 2.5
            source_count = int(np.count_nonzero(lat_mask) * np.count_nonzero(lon_mask))
            if source_count != 400:
                raise ValueError(
                    f"Cell ({target_latitude}, {target_longitude}) expected "
                    f"400 source samples, got {source_count}"
                )
            output[lat_index, lon_index] = float(latitude_values[:, lon_mask].mean())

    if not np.isfinite(output).all():
        raise ValueError("Generated MOLA elevation contains NaN or infinity")
    return output


def build_asset(image_path: Path, label_path: Path, output_path: Path) -> None:
    _validate_source(image_path, label_path)
    source = np.fromfile(image_path, dtype=">i2")
    if source.size != SOURCE_SHAPE[0] * SOURCE_SHAPE[1]:
        raise ValueError(
            f"Decoded MOLA sample count expected {SOURCE_SHAPE[0] * SOURCE_SHAPE[1]}, "
            f"got {source.size}"
        )
    elevation = _aggregate_5_degree_grid(
        source.reshape(SOURCE_SHAPE).astype(np.float32)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(str(output_path), "w", format="NETCDF4") as dataset:
        dataset.createDimension("latitude", len(TARGET_LATITUDE))
        dataset.createDimension("longitude", len(TARGET_LONGITUDE))
        latitude = dataset.createVariable("latitude", "f4", ("latitude",))
        longitude = dataset.createVariable("longitude", "f4", ("longitude",))
        elevation_variable = dataset.createVariable(
            "elevation",
            "f4",
            ("latitude", "longitude"),
            zlib=True,
            complevel=4,
        )
        latitude.units = "degree_north"
        longitude.units = "degree_east"
        elevation_variable.units = "meter"
        latitude[:] = TARGET_LATITUDE
        longitude[:] = TARGET_LONGITUDE
        elevation_variable[:] = elevation

        dataset.title = "AstraAtmos MOLA global topography at 5 degree resolution"
        for name, value in REQUIRED_PROVENANCE.items():
            dataset.setncattr(name, value)
        dataset.generated_at_utc = datetime.now(timezone.utc).isoformat()


def validate_asset(path: Path) -> None:
    elevation, latitude, longitude = load_mola_asset(path)
    if elevation.shape != (36, 72):
        raise ValueError(
            f"Platform asset elevation.shape expected (36, 72), "
            f"got {elevation.shape}"
        )
    if not np.array_equal(latitude.astype(np.float32), TARGET_LATITUDE):
        raise ValueError(
            "Platform asset latitude expected "
            f"{TARGET_LATITUDE.tolist()!r}, got {latitude.tolist()!r}"
        )
    if not np.array_equal(longitude.astype(np.float32), TARGET_LONGITUDE):
        raise ValueError(
            "Platform asset longitude expected "
            f"{TARGET_LONGITUDE.tolist()!r}, got {longitude.tolist()!r}"
        )
    with netCDF4.Dataset(str(path)) as dataset:
        attribute_names = set(dataset.ncattrs())
        for field, expected in REQUIRED_PROVENANCE.items():
            actual = dataset.getncattr(field) if field in attribute_names else "<missing>"
            if actual != expected:
                raise ValueError(
                    f"Platform asset {field} expected {expected!r}, got {actual!r}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    if args.validate:
        validate_asset(args.output)
        print(f"Validated MOLA asset: {args.output}")
        return

    source_dir = args.source_dir or (
        Path(tempfile.gettempdir()) / "astraatmos_mola_megdr_source"
    )
    image_path, label_path = _ensure_sources(source_dir)
    build_asset(image_path, label_path, args.output)
    validate_asset(args.output)
    print(f"Built MOLA asset: {args.output}")


if __name__ == "__main__":
    main()
