from pathlib import Path

from services.mcd_file_locator import find_mcd_files


def test_find_mcd_files_matches_year_case_insensitively(tmp_path: Path):
    my27 = tmp_path / "MCD_MY27_Lat-90-90_real.nc"
    my28 = tmp_path / "mcd_my28_real.NC"
    unscoped = tmp_path / "unscoped.nc"
    for path in (my27, my28, unscoped):
        path.touch()

    assert find_mcd_files(tmp_path, 27) == [my27]
    assert find_mcd_files(tmp_path, 28) == [my28]
    assert find_mcd_files(tmp_path, 29) == []
