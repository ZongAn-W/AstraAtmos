"""Scientific and publication-contract checks for the standalone 3-hour builder."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date, timedelta
from pathlib import Path

import netCDF4
import numpy as np
import pytest

import scripts.build_earth_merra2_3hourly as builder
import scripts.verify_earth_merra2_3hourly as verifier
import services.earth_dataset_metadata as earth_metadata
from scripts.build_earth_merra2_3hourly import (
    DATASET_ID,
    DATA_FILE,
    SCHEMA,
    build_dataset,
    conservative_regrid,
    dataset_fingerprint,
    discover_source_files,
    target_grid,
    temporal_mean_three_hours,
    verify_existing_release,
)
from scripts.verify_earth_merra2_3hourly import verify_package


VARIABLE_UNITS = {
    "TO3": "Dobsons",
    "U10M": "m s-1",
    "V10M": "m s-1",
    "T2M": "K",
    "SWGDN": "W m-2",
}
SOURCE_LAT = np.arange(-90.0, 91.0, 30.0)
SOURCE_LON = np.arange(-180.0, 180.0, 30.0)


def make_source_archive(root: Path, *, bad: str | None = None) -> Path:
    """Seven small global source days with deliberately unequal hourly values."""
    for day_index in range(7):
        day = date(2020, 1, 1) + timedelta(days=day_index)
        for product, names in (
            ("slv", ("TO3", "U10M", "V10M", "T2M")),
            ("rad", ("SWGDN",)),
        ):
            if bad == "missing_pair" and day_index == 0 and product == "rad":
                continue
            directory = root / product
            directory.mkdir(parents=True, exist_ok=True)
            named_product = "rad" if bad == "wrong_product" and day_index == 0 and product == "slv" else product
            path = directory / f"MERRA2_400.tavg1_2d_{named_product}_Nx.{day:%Y%m%d}.nc4"
            with netCDF4.Dataset(path, "w") as source:
                time_count = 23 if bad == "incomplete_hours" and day_index == 0 and product == "slv" else 24
                source.createDimension("time", time_count)
                source.createDimension("lat", len(SOURCE_LAT))
                source.createDimension("lon", len(SOURCE_LON))
                time_var = source.createVariable("time", "f8", ("time",))
                time_var.units = f"hours since {day.isoformat()} 00:00:00"
                time_var.calendar = "standard"
                time_values = np.arange(time_count, dtype=np.float64) + 0.5
                if day_index == 0 and product == "slv":
                    if bad == "duplicate_hour":
                        time_values[2] = time_values[1]
                    elif bad == "offset_hour":
                        time_values += 0.5
                time_var[:] = time_values
                source.createVariable("lat", "f8", ("lat",))[:] = SOURCE_LAT
                source.createVariable("lon", "f8", ("lon",))[:] = SOURCE_LON
                hourly = np.repeat(np.arange(8) * 10.0, 3) + np.tile([1.0, 4.0, 13.0], 8)
                hourly += day_index * 100.0
                for name in names:
                    variable = source.createVariable(name, "f4", ("time", "lat", "lon"), fill_value=-9999.0)
                    variable.units = VARIABLE_UNITS[name]
                    variable[:] = hourly[:time_count, None, None] + list(VARIABLE_UNITS).index(name)
                    if day_index == 0 and name == "TO3":
                        if bad == "unit":
                            variable.units = "kg m-2"
                        elif bad == "missing":
                            variable[0, 3, 0] = -9999.0
                        elif bad == "nan":
                            variable[0, 3, 0] = np.nan
                if day_index == 0 and product == "rad" and bad == "mismatched_product_hours":
                    time_var[:] = time_values + 0.25
    return root


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_target_centres_cover_exact_global_cell_bounds():
    lat, lon = target_grid()
    assert (len(lat), len(lon)) == (240, 480)
    np.testing.assert_allclose(np.diff(lat), 0.75, rtol=0, atol=0)
    np.testing.assert_allclose(np.diff(lon), 0.75, rtol=0, atol=0)
    np.testing.assert_array_equal(lat[[0, -1]], [-89.625, 89.625])
    np.testing.assert_array_equal(lon[[0, -1]], [-179.625, 179.625])
    assert (lat[0] - 0.375, lat[-1] + 0.375) == (-90.0, 90.0)
    assert (lon[0] - 0.375, lon[-1] + 0.375) == (-180.0, 180.0)


def test_three_hours_are_a_true_mean_and_require_complete_support():
    hourly = np.array([[1.0, 0.0], [4.0, 0.0], [13.0, 0.0], [20.0, 2.0], [22.0, 4.0], [30.0, 6.0]])
    mean = temporal_mean_three_hours(hourly)
    np.testing.assert_allclose(mean, [[6.0, 0.0], [24.0, 4.0]])
    assert mean[0, 0] != hourly[0, 0]  # A stride of three would fail this check.
    valid = np.ones(hourly.shape, dtype=bool)
    valid[1, 0] = False
    incomplete = temporal_mean_three_hours(hourly, valid)
    assert np.isnan(incomplete[0, 0])
    assert incomplete[0, 1] == 0.0  # A measured zero remains a measured zero.
    np.testing.assert_allclose(incomplete[1], [24.0, 4.0])
    masked = np.ma.array(hourly, mask=~valid)
    np.testing.assert_allclose(temporal_mean_three_hours(masked), incomplete, equal_nan=True)
    with pytest.raises(ValueError, match="multiple of three"):
        temporal_mean_three_hours(hourly[:5])


def test_conservative_regrid_preserves_constants_and_spherical_global_mean():
    constant = conservative_regrid(np.full((7, 12), 7.0), SOURCE_LAT, SOURCE_LON)
    np.testing.assert_allclose(constant, 7.0, rtol=0, atol=2e-14)
    field = np.arange(7 * 12, dtype=np.float64).reshape(7, 12) ** 2
    mapped = conservative_regrid(field, SOURCE_LAT, SOURCE_LON)
    source_edges = np.r_[-90.0, (SOURCE_LAT[:-1] + SOURCE_LAT[1:]) / 2.0, 90.0]
    source_area = np.diff(np.sin(np.deg2rad(source_edges)))
    target_area = np.diff(np.sin(np.deg2rad(np.arange(241) * 0.75 - 90.0)))
    source_mean = np.sum(field * source_area[:, None]) / (2.0 * len(SOURCE_LON))
    target_mean = np.sum(mapped * target_area[:, None]) / (2.0 * 480)
    assert target_mean == pytest.approx(source_mean, rel=1e-12)


def test_periodic_native_seam_contributes_at_both_target_edges():
    seam = np.zeros((7, 12))
    seam[:, 0] = 1.0
    mapped = conservative_regrid(seam, SOURCE_LAT, SOURCE_LON)
    np.testing.assert_allclose(mapped[:, :20], 1.0)
    np.testing.assert_allclose(mapped[:, -20:], 1.0)
    np.testing.assert_allclose(mapped[:, 20:-20], 0.0)


@pytest.mark.parametrize("kind", ["mask", "nan", "infinity", "sentinel"])
def test_missing_regrid_support_is_nan_including_both_periodic_edges(kind):
    values = np.full((7, 12), 9.0)
    valid = np.ones(values.shape, dtype=bool)
    if kind == "mask":
        valid[3, 0] = False
    else:
        values[3, 0] = {"nan": np.nan, "infinity": np.inf, "sentinel": 1e15}[kind]
    mapped = conservative_regrid(values, SOURCE_LAT, SOURCE_LON, valid)
    assert np.isnan(mapped[100:140, :20]).all()
    assert np.isnan(mapped[100:140, -20:]).all()
    assert np.isfinite(mapped).any()
    np.testing.assert_allclose(mapped[np.isfinite(mapped)], 9.0)
    assert not np.any(mapped == 0.0)


def test_streaming_smoke_publication_masks_identity_and_reuse(tmp_path):
    raw = make_source_archive(tmp_path / "raw", bad="missing")
    # chm's presence is irrelevant: the builder must read the paired slv/rad archive.
    (raw / "chm").mkdir()
    (raw / "chm" / "unusable.nc4").write_bytes(b"not a NetCDF file")
    output = build_dataset(raw, tmp_path / "release", smoke_days=7)
    manifest_path = output.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert output.name == DATA_FILE
    assert manifest["dataset_id"] == DATASET_ID
    assert manifest["build_mode"] == "smoke"
    assert manifest["dimensions"] == {"time": 56, "lat": 240, "lon": 480, "bounds": 2}
    assert manifest["frequency_hours"] == 3
    assert manifest["time_start"] == "2020-01-01T01:30:00Z"
    assert manifest["time_end"] == "2020-01-07T22:30:00Z"
    assert manifest["processing"]["spatial_method"] == "spherical_area_weighted_overlap"
    assert manifest["processing"]["temporal_method"] == "mean_of_three_complete_hourly_samples"
    assert manifest["training_profile"]["window"] == 56
    assert manifest["training_profile"]["horizon"] == 24
    assert manifest["data_sha256"] == sha256(output)
    assert len(manifest["source_files"]) == 14
    assert {item["product"] for item in manifest["source_files"]} == {"slv", "rad"}
    assert manifest["fingerprint_rule"] == 2
    assert manifest["source_inventory_version"] == 3
    # mtime_ns is provenance, not identity: it left the inventory entries for the
    # top-level block that no identity digest covers.
    assert all({"date", "product", "path", "name", "bytes", "sha256"} <= item.keys()
               for item in manifest["source_files"])
    assert not any("mtime_ns" in item for item in manifest["source_files"])
    assert set(manifest["source_mtimes"]) == {item["path"] for item in manifest["source_files"]}
    assert manifest["source_sha256"] == hashlib.sha256(
        builder.canonical_json(builder.source_identity_projection(manifest["source_files"]))
    ).hexdigest()
    assert "full-file-content-SHA256" in manifest["source_fingerprint_basis"]
    for item in manifest["source_files"]:
        assert item["sha256"] == sha256(raw / item["path"])
        assert manifest["source_mtimes"][item["path"]] == (raw / item["path"]).stat().st_mtime_ns
    canonical_manifest = json.dumps(
        {key: value for key, value in manifest.items()
         if key not in ("dataset_fingerprint", "source_mtimes", "fingerprint_rule")}
        | {"source_files": builder.source_identity_projection(manifest["source_files"])},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    identity = {"dataset_id": DATASET_ID, "dataset_version": "v1",
                "manifest_sha256": hashlib.sha256(canonical_manifest).hexdigest(),
                "data_sha256": sha256(output)}
    expected_fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert manifest["dataset_fingerprint"] == expected_fingerprint
    with netCDF4.Dataset(output) as stored:
        stored.set_auto_mask(False)
        assert stored.variables["TO3"].shape == (56, 240, 480)
        np.testing.assert_array_equal(stored.variables["time"][:], np.arange(56) * 3 + 1.5)
        np.testing.assert_array_equal(stored.variables["time_bounds"][[0, -1]], [[0.0, 3.0], [165.0, 168.0]])
        np.testing.assert_array_equal(stored.variables["lat_bounds"][[0, -1]], [[-90.0, -89.25], [89.25, 90.0]])
        np.testing.assert_array_equal(stored.variables["lon_bounds"][[0, -1]], [[-180.0, -179.25], [179.25, 180.0]])
        for name, expected_unit in zip(VARIABLE_UNITS, ["DU", "m s-1", "m s-1", "K", "W m-2"]):
            assert stored.variables[name].units == expected_unit
            assert manifest["variables"][name]["units"] == expected_unit
            field = stored.variables[name][:]
            mask = np.asarray(stored.variables[f"{name}_valid_mask"][:])
            assert set(np.unique(mask)).issubset({0, 1})
            np.testing.assert_array_equal(mask == 1, np.isfinite(field))
            assert manifest["variables"][name]["target_missing"] == int((mask == 0).sum())
            assert manifest["variables"][name]["missing_rate"] == pytest.approx(float((mask == 0).mean()))
            expected = np.repeat(np.arange(7) * 100.0, 8) + np.tile(np.arange(8) * 10.0 + 6.0, 7)
            expected += list(VARIABLE_UNITS).index(name)
            np.testing.assert_allclose(field[:, 0, 100], expected)
        assert np.isnan(stored.variables["TO3"][0, 100:140, :20]).all()
        assert np.isnan(stored.variables["TO3"][0, 100:140, -20:]).all()
    before = (sha256(output), sha256(manifest_path), output.stat().st_mtime_ns, manifest_path.stat().st_mtime_ns)
    assert build_dataset(raw, output.parent, smoke_days=7) == output
    assert before == (sha256(output), sha256(manifest_path), output.stat().st_mtime_ns, manifest_path.stat().st_mtime_ns)
    # An existing publication with damaged bytes is retained and rejected.
    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / DATA_FILE).write_bytes(output.read_bytes() + b"changed bytes")
    (corrupt / "manifest.json").write_bytes(manifest_path.read_bytes())
    with pytest.raises(ValueError, match="SHA-256"):
        verify_existing_release(corrupt)
    assert (corrupt / DATA_FILE).read_bytes().endswith(b"changed bytes")
    manifest["time_zone"] = "changed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        build_dataset(raw, output.parent, smoke_days=7)
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["time_zone"] == "changed"


def test_three_hour_source_inventory_and_canonical_json_are_reproducible(tmp_path):
    raw = make_source_archive(tmp_path / "raw")
    days = [date(2020, 1, 1) + timedelta(days=index) for index in range(7)]
    index = discover_source_files(raw, days)

    first, _, _ = builder._source_inventory(index, raw.resolve())
    second, _, _ = builder._source_inventory(index, raw.resolve())

    assert first == second
    assert builder.canonical_json(first) == builder.canonical_json(second)
    assert hashlib.sha256(builder.canonical_json(first)).digest() == hashlib.sha256(
        builder.canonical_json(second)
    ).digest()
    assert all(len(item["sha256"]) == 64 for item in first)


def test_changed_source_content_cannot_reuse_existing_output(tmp_path):
    raw = make_source_archive(tmp_path / "raw")
    output = build_dataset(raw, tmp_path / "release", smoke_days=7)
    manifest_path = output.with_name("manifest.json")
    previous_output = sha256(output)
    previous_manifest = sha256(manifest_path)
    target = raw / "slv" / "MERRA2_400.tavg1_2d_slv_Nx.20200101.nc4"
    before = target.stat()
    contents = bytearray(target.read_bytes())
    contents[len(contents) // 2] ^= 1
    target.write_bytes(contents)
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert target.stat().st_size == before.st_size
    assert target.stat().st_mtime_ns == before.st_mtime_ns

    with pytest.raises(ValueError, match="source inventory does not match"):
        build_dataset(raw, output.parent, smoke_days=7)

    assert sha256(output) == previous_output
    assert sha256(manifest_path) == previous_manifest


@pytest.mark.parametrize("change_size", [False, True], ids=["same-size-restored-mtime", "size-changed"])
def test_build_rejects_source_content_change_before_manifest(tmp_path, monkeypatch, change_size):
    raw = make_source_archive(tmp_path / "raw")
    target = raw.resolve() / "rad" / "MERRA2_400.tavg1_2d_rad_Nx.20200107.nc4"
    original_read = builder._read_hourly
    changed = False

    def read_then_change(path, *args, **kwargs):
        nonlocal changed
        result = original_read(path, *args, **kwargs)
        if path == target and not changed:
            before = target.stat()
            contents = bytearray(target.read_bytes())
            if change_size:
                contents.extend(b"x")
            else:
                contents[-1] ^= 1
            target.write_bytes(contents)
            if not change_size:
                os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
                after = target.stat()
                assert after.st_size == before.st_size
                assert after.st_mtime_ns == before.st_mtime_ns
            changed = True
        return result

    monkeypatch.setattr(builder, "_read_hourly", read_then_change)
    output_dir = tmp_path / "release"
    with pytest.raises(ValueError, match="source archive .* changed"):
        build_dataset(raw, output_dir, smoke_days=7)

    assert changed
    assert (output_dir / DATA_FILE).exists()
    assert not (output_dir / "manifest.json").exists()


def test_build_allows_mtime_only_change_when_source_content_digest_matches(tmp_path, monkeypatch):
    raw = make_source_archive(tmp_path / "raw")
    target = raw.resolve() / "rad" / "MERRA2_400.tavg1_2d_rad_Nx.20200107.nc4"
    original_read = builder._read_hourly
    changed = False

    def read_then_touch(path, *args, **kwargs):
        nonlocal changed
        result = original_read(path, *args, **kwargs)
        if path == target and not changed:
            before = target.stat()
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
            assert target.stat().st_mtime_ns != before.st_mtime_ns
            changed = True
        return result

    monkeypatch.setattr(builder, "_read_hourly", read_then_touch)
    output_dir = tmp_path / "release"
    output = build_dataset(raw, output_dir, smoke_days=7)

    manifest = json.loads(output.with_name("manifest.json").read_text(encoding="utf-8"))
    assert changed
    assert output.exists()
    assert manifest["source_sha256"] == hashlib.sha256(
        builder.canonical_json(builder.source_identity_projection(manifest["source_files"]))
    ).hexdigest()
    entry = next(item for item in manifest["source_files"] if item["name"] == target.name)
    assert entry["sha256"] == sha256(target)
    # The manifest keeps the provenance recorded before the build; identity and
    # source digest are unaffected by the later timestamp drift.
    assert manifest["source_mtimes"][entry["path"]] != target.stat().st_mtime_ns


def test_verify_existing_release_accepts_pure_mtime_drift_without_moving_identity(tmp_path):
    """Touching every source file must neither rebuild nor re-identify the release."""
    raw = make_source_archive(tmp_path / "raw")
    output = build_dataset(raw, tmp_path / "release", smoke_days=7)
    manifest_path = output.with_name("manifest.json")
    published = manifest_path.read_bytes()
    manifest = json.loads(published.decode("utf-8"))
    for path in raw.rglob("*.nc4"):
        before = path.stat()
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 3_000_000_000))
    assert all((raw / item["path"]).stat().st_mtime_ns != manifest["source_mtimes"][item["path"]]
               for item in manifest["source_files"])

    assert build_dataset(raw, output.parent, smoke_days=7) == output
    # The publication is reused unchanged: same bytes, same identity, same digest.
    assert manifest_path.read_bytes() == published
    reused = verify_existing_release(output.parent)
    assert reused["dataset_fingerprint"] == manifest["dataset_fingerprint"]
    assert reused["source_sha256"] == manifest["source_sha256"]


# Minimal manifest fixture for the cross-implementation golden vector. The three
# implementations (builder, service, independent verifier) are deliberately
# separate, so this locks them to one frozen expectation.
GOLDEN_SOURCE_FILES = (
    {"date": "2020-01-01", "product": "slv", "path": "slv/MERRA2.slv.20200101.nc4",
     "name": "MERRA2.slv.20200101.nc4", "bytes": 1024, "sha256": "a" * 64},
    {"date": "2020-01-01", "product": "rad", "path": "rad/MERRA2.rad.20200101.nc4",
     "name": "MERRA2.rad.20200101.nc4", "bytes": 2048, "sha256": "b" * 64},
    {"date": "2020-01-02", "product": "slv", "path": "slv/MERRA2.slv.20200102.nc4",
     "name": "MERRA2.slv.20200102.nc4", "bytes": 4096, "sha256": "c" * 64},
    {"date": "2020-01-02", "product": "rad", "path": "rad/MERRA2.rad.20200102.nc4",
     "name": "MERRA2.rad.20200102.nc4", "bytes": 8192, "sha256": "d" * 64},
)
GOLDEN_IDENTITY_FIELDS = ("date", "product", "path", "bytes", "sha256")
GOLDEN_LEGACY_SOURCE_SHA = "38e107533f22e45033fb6d0cbf057ecba6fb3d2a2af242edaefd2206d7784beb"
GOLDEN_CONTENT_SOURCE_SHA = "16aa682afb93b5a7f9f5965c7a5e95a47b1dbf12c630b706f9523770ee958935"
GOLDEN_RULE1_CONTENT_SHA = "11af7ab3c960d88f6e205b83e80ab21d903ca1014b999fbb9278b486a37438cf"
GOLDEN_RULE2_CONTENT_SHA = "160f0857e0f750d321faa7c214f03fe80d8bb676f05451f6ee35f4470fb846f6"


def _golden_canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def golden_manifests():
    """One legacy rule 1 fixture and one content rule 2 fixture of the same data."""
    identity = [{key: item[key] for key in GOLDEN_IDENTITY_FIELDS} for item in GOLDEN_SOURCE_FILES]
    legacy_files = [dict(item, mtime_ns=1_600_000_000_000_000_000 + index)
                    for index, item in enumerate(GOLDEN_SOURCE_FILES)]
    base = {"schema": SCHEMA, "planet": "Earth", "dataset_id": DATASET_ID,
            "dataset_version": "v1", "data_file": DATA_FILE,
            "data_bytes": 4096, "data_sha256": "e" * 64}
    legacy = dict(base, source_files=legacy_files,
                  source_sha256=hashlib.sha256(_golden_canonical(legacy_files)).hexdigest())
    content = dict(base, source_files=[dict(item) for item in GOLDEN_SOURCE_FILES],
                   source_sha256=hashlib.sha256(_golden_canonical(identity)).hexdigest(),
                   source_mtimes={item["path"]: item["bytes"] for item in GOLDEN_SOURCE_FILES},
                   source_inventory_version=3, fingerprint_rule=2)
    legacy["dataset_fingerprint"] = dataset_fingerprint(legacy)
    content["dataset_fingerprint"] = dataset_fingerprint(content)
    return legacy, content


def test_identity_rule_implementations_agree_on_a_golden_vector():
    """Builder, service and independent verifier must digest both rules identically."""
    legacy, content = golden_manifests()
    assert legacy["source_sha256"] == GOLDEN_LEGACY_SOURCE_SHA
    assert content["source_sha256"] == GOLDEN_CONTENT_SOURCE_SHA
    assert manifest_content_sha256_paths(legacy) == [GOLDEN_RULE1_CONTENT_SHA] * 3
    assert manifest_content_sha256_paths(content) == [GOLDEN_RULE2_CONTENT_SHA] * 3
    assert earth_metadata.threehour_source_inventory_sha256(content["source_files"], 3) == GOLDEN_CONTENT_SOURCE_SHA
    assert verifier._source_inventory_sha256(content["source_files"], 3) == GOLDEN_CONTENT_SOURCE_SHA
    assert earth_metadata.threehour_source_inventory_sha256(legacy["source_files"], 2) == GOLDEN_LEGACY_SOURCE_SHA
    assert verifier._source_inventory_sha256(legacy["source_files"], 2) == GOLDEN_LEGACY_SOURCE_SHA
    # The identity wrapper reuses the same digest and must reproduce the frozen field.
    for manifest in (legacy, content):
        assert earth_metadata.threehour_release_identity(manifest)["dataset_fingerprint"] == \
            manifest["dataset_fingerprint"]


def manifest_content_sha256_paths(manifest):
    """The three independent digests for one manifest, in a fixed order."""
    return [builder.manifest_content_sha256(manifest),
            earth_metadata.threehour_manifest_content_sha256(manifest),
            verifier._manifest_content_sha256(manifest)]


def test_final_prepublication_digest_failure_does_not_create_manifest(tmp_path, monkeypatch):
    raw = make_source_archive(tmp_path / "raw")
    target = raw.resolve() / "slv" / "MERRA2_400.tavg1_2d_slv_Nx.20200101.nc4"
    original_assert = builder._assert_source_digests
    checks = 0

    def mutate_after_build_check(index, source_root, expected):
        nonlocal checks
        checks += 1
        original_assert(index, source_root, expected)
        if checks == 1:
            before = target.stat()
            contents = bytearray(target.read_bytes())
            contents[len(contents) // 2] ^= 1
            target.write_bytes(contents)
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
            after = target.stat()
            assert after.st_size == before.st_size
            assert after.st_mtime_ns == before.st_mtime_ns

    monkeypatch.setattr(builder, "_assert_source_digests", mutate_after_build_check)
    output_dir = tmp_path / "release"
    with pytest.raises(ValueError, match="source archive content changed during build"):
        build_dataset(raw, output_dir, smoke_days=7)

    assert checks == 2
    assert (output_dir / DATA_FILE).exists()
    assert not (output_dir / "manifest.json").exists()


@pytest.mark.parametrize("bad,match", [("duplicate_hour", "time"), ("offset_hour", "time"),
                                       ("incomplete_hours", "time"), ("mismatched_product_hours", "time"),
                                       ("unit", "unit")])
def test_malformed_hourly_sources_never_publish_a_manifest(tmp_path, bad, match):
    raw = make_source_archive(tmp_path / "raw", bad=bad)
    before = {path: sha256(path) for path in raw.rglob("*.nc4")}
    with pytest.raises(ValueError, match=match):
        build_dataset(raw, tmp_path / "release", smoke_days=7)
    assert not (tmp_path / "release" / "manifest.json").exists()
    assert all(sha256(path) == digest for path, digest in before.items())


@pytest.mark.parametrize("bad,match", [("missing_pair", "missing slv/rad"), ("wrong_product", "product")])
def test_selected_source_products_must_be_complete_and_correct(tmp_path, bad, match):
    raw = make_source_archive(tmp_path / "raw", bad=bad)
    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(7)]
    with pytest.raises(ValueError, match=match):
        discover_source_files(raw, days)


def test_duplicate_selected_source_file_is_rejected(tmp_path):
    raw = make_source_archive(tmp_path / "raw")
    source = next((raw / "slv").glob("*20200101*"))
    source.with_name(source.name.replace("400", "401")).write_bytes(source.read_bytes())
    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(7)]
    with pytest.raises(ValueError, match="duplicate"):
        discover_source_files(raw, days)


def test_partial_existing_output_is_retained_without_overwrite(tmp_path):
    raw = make_source_archive(tmp_path / "raw")
    output_dir = tmp_path / "release"
    output_dir.mkdir()
    existing = output_dir / DATA_FILE
    existing.write_bytes(b"existing partial output")
    with pytest.raises(ValueError, match="partial"):
        build_dataset(raw, output_dir, smoke_days=7)
    assert existing.read_bytes() == b"existing partial output"
    assert not existing.with_name("manifest.json").exists()


@pytest.mark.parametrize("smoke_days", [6, 31])
def test_smoke_duration_is_limited_to_seven_through_thirty_days(tmp_path, smoke_days):
    with pytest.raises(ValueError, match="between 7 and 30"):
        build_dataset(tmp_path / "missing", tmp_path / "release", smoke_days=smoke_days)


def test_independent_verifier_agrees_on_cf_packing_valid_range_and_du(tmp_path):
    raw = make_source_archive(tmp_path / "raw")
    for path in (raw / "slv").glob("*.nc4"):
        with netCDF4.Dataset(path, "a") as source:
            ozone = source["TO3"]
            ozone.units = "DU"
            ozone.scale_factor = np.float32(.25)
            ozone.add_offset = np.float32(3)
            # The range acts on stored samples, before packing is decoded.
            ozone.valid_range = np.array([0, 700], dtype="float32")
    output = build_dataset(raw, tmp_path / "release", smoke_days=7)
    with netCDF4.Dataset(output) as release:
        # First three stored samples are 1,4,13: mean(0.25*x+3)=4.5.
        np.testing.assert_allclose(release["TO3"][0], 4.5)
    report = verify_package(output.parent, raw_root=raw)
    assert report["validation"] == "passed"
    assert report["raw_comparison"]["samples_checked"] == 405
    # The independent verifier reports the rule it applied and keeps the two
    # frozen identity digests in its proof output.
    manifest = json.loads((output.parent / "manifest.json").read_text(encoding="utf-8"))
    assert report["fingerprint_rule"] == 2
    assert report["source_inventory_version"] == 3
    assert report["manifest_content_sha256"] == verifier._manifest_content_sha256(manifest)
    assert report["dataset_fingerprint"] == manifest["dataset_fingerprint"]
    assert report["manifest_file_sha256"] == sha256(output.with_name("manifest.json"))


@pytest.mark.parametrize("change,match", [("time", "time has duplicates"),
                                        ("mask", "mask/value mismatch"),
                                        ("statistics", "statistics mismatch")])
def test_independent_verifier_rejects_rehashed_but_invalid_content(tmp_path, change, match):
    raw = make_source_archive(tmp_path / "raw")
    output = build_dataset(raw, tmp_path / "release", smoke_days=7)
    manifest_path = output.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if change == "statistics":
        manifest["variables"]["TO3"]["missing_rate"] = .5
    else:
        with netCDF4.Dataset(output, "a") as release:
            if change == "time":
                release["time"][1] = release["time"][0]
            else:
                release["TO3_valid_mask"][0, 120, 240] = 0
        manifest["data_bytes"] = output.stat().st_size
        manifest["data_sha256"] = sha256(output)
    manifest["dataset_fingerprint"] = dataset_fingerprint(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = sha256(output), sha256(manifest_path)
    with pytest.raises(ValueError, match=match):
        verify_package(output.parent)
    assert (sha256(output), sha256(manifest_path)) == before


def test_independent_verifier_rejects_data_sha_corruption(tmp_path):
    raw = make_source_archive(tmp_path / "raw")
    output = build_dataset(raw, tmp_path / "release", smoke_days=7)
    manifest_path = output.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["data_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_package(output.parent)
