from pathlib import Path

from config import resolve_mcd_dir


def test_resolve_mcd_dir_prefers_explicit_configuration(tmp_path: Path):
    configured = tmp_path / "external-mcd"

    assert resolve_mcd_dir(tmp_path / "data", configured) == configured


def test_resolve_mcd_dir_prefers_canonical_lowercase_directory(tmp_path: Path):
    data_dir = tmp_path / "data"
    canonical = data_dir / "mcd"
    canonical.mkdir(parents=True)

    assert resolve_mcd_dir(data_dir, None) == canonical


def test_resolve_mcd_dir_falls_back_to_legacy_uppercase_directory(tmp_path: Path):
    data_dir = tmp_path / "data"
    legacy = data_dir / "MCD"
    legacy.mkdir(parents=True)

    assert resolve_mcd_dir(data_dir, None) == legacy


def test_resolve_mcd_dir_returns_canonical_path_when_data_is_missing(tmp_path: Path):
    data_dir = tmp_path / "missing-data"

    assert resolve_mcd_dir(data_dir, None) == data_dir / "mcd"


def test_gitignore_covers_both_mcd_directory_spellings():
    repo_root = Path(__file__).resolve().parents[3]
    rules = (repo_root / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "AstraAtmos_backend/backend/data/mcd/" in rules
    assert "AstraAtmos_backend/backend/data/MCD/" in rules
