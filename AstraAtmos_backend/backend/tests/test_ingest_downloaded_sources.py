import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scripts.ingest_downloaded_sources import (  # noqa: E402
    _default_source_root,
    require_nomad_start_my,
    sync_openmars_files,
)


def test_default_source_root_prefers_workspace_data_directory():
    expected = Path(__file__).resolve().parents[4] / "Data"
    assert _default_source_root() == expected


def test_sync_openmars_files_copies_only_nc_files(tmp_path):
    source = tmp_path / "source_openmars"
    target = tmp_path / "target_openmars"
    source.mkdir()
    (source / "openmars_ozo_my27_ls2_my27_ls17.nc").write_bytes(b"netcdf-bytes")
    (source / "OpenMARS-ozone-reference-manual.pdf").write_bytes(b"manual")

    copied = sync_openmars_files(source, target)

    assert [path.name for path in copied] == ["openmars_ozo_my27_ls2_my27_ls17.nc"]
    assert (target / "openmars_ozo_my27_ls2_my27_ls17.nc").read_bytes() == b"netcdf-bytes"
    assert not (target / "OpenMARS-ozone-reference-manual.pdf").exists()


def test_require_nomad_start_my_fails_for_nomad_ingest_without_start_year():
    with pytest.raises(ValueError, match="--nomad-start-my"):
        require_nomad_start_my(only="nomad", nomad_start_my=None)


def test_require_nomad_start_my_allows_non_nomad_ingest_without_start_year():
    assert require_nomad_start_my(only="openmars", nomad_start_my=None) is None
