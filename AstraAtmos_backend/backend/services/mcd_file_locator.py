from __future__ import annotations

import re
from pathlib import Path


_MARS_YEAR_PATTERN = re.compile(
    r"(?:^|[^A-Za-z0-9])MY(?P<year>\d+)", re.IGNORECASE
)


def _mars_year_from_name(file_name: str) -> int | None:
    match = _MARS_YEAR_PATTERN.search(file_name)
    return int(match.group("year")) if match else None


def find_mcd_files(mcd_dir: Path, mars_year: int) -> list[Path]:
    directory = Path(mcd_dir)
    if not directory.is_dir():
        return []

    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".nc"
            and _mars_year_from_name(path.name) == int(mars_year)
        ),
        key=lambda path: path.name.casefold(),
    )
