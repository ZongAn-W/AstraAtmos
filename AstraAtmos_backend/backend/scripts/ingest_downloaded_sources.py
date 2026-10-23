from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import OVERVIEW_OZONE_MATCH_TOLERANCE_LS, SUPPORTED_MARS_YEARS  # noqa: E402
from scripts.build_mcd_overview_dataset import build_year  # noqa: E402
from scripts.build_nomad_gridded_dataset import build_nomad_gridded_datasets  # noqa: E402


def _default_source_root() -> Path:
    candidates = [
        WORKSPACE_ROOT / "Data",
        REPO_ROOT / "Data",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def sync_openmars_files(source_dir: Path, target_dir: Path) -> list[Path]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"OpenMARS source directory not found: {source_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source_path in sorted(source_dir.glob("*.nc")):
        target_path = target_dir / source_path.name
        shutil.copy2(source_path, target_path)
        copied.append(target_path)
    return copied


def require_nomad_start_my(only: str, nomad_start_my: int | None) -> int | None:
    if only in {"all", "nomad"} and nomad_start_my is None:
        raise ValueError("--nomad-start-my is required when ingesting NOMAD")
    return nomad_start_my


def ingest_mcd(source_root: Path, backend_data: Path, years: list[int]) -> list[Path]:
    reference_dir = source_root / "MCD_Output_global_10m_ls_lst"
    output_dir = backend_data / "mcd_overview"
    base_mcd_dir = backend_data / "mcd"
    written: list[Path] = []
    for year in years:
        written.append(build_year(year, reference_dir, output_dir, base_mcd_dir=base_mcd_dir, mode="reference"))
    return written


def ingest_openmars(source_root: Path, backend_data: Path) -> list[Path]:
    return sync_openmars_files(source_root / "openmars", backend_data / "openmars")


def ingest_nomad(
    source_root: Path,
    backend_data: Path,
    years: list[int],
    nomad_start_my: int,
    tolerance_ls: float,
) -> list[Path]:
    return build_nomad_gridded_datasets(
        nomad_csv_path=source_root / "NOMAD" / "UVIS_ozone_column_retrieval_dataset_v1.txt",
        mcd_overview_dir=backend_data / "mcd_overview",
        output_dir=backend_data / "nomad",
        start_my=nomad_start_my,
        years=years,
        tolerance_ls=tolerance_ls,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest downloaded AstraAtmos data sources into backend/data.")
    parser.add_argument("--source-root", type=Path, default=_default_source_root())
    parser.add_argument("--backend-data", type=Path, default=BACKEND_DIR / "data")
    parser.add_argument("--only", choices=["all", "mcd", "openmars", "nomad"], default="all")
    parser.add_argument("--years", nargs="+", type=int, default=SUPPORTED_MARS_YEARS)
    parser.add_argument("--nomad-start-my", type=int, default=None)
    parser.add_argument("--nomad-tolerance-ls", type=float, default=OVERVIEW_OZONE_MATCH_TOLERANCE_LS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_nomad_start_my(args.only, args.nomad_start_my)

    if args.only in {"all", "mcd"}:
        for path in ingest_mcd(args.source_root, args.backend_data, args.years):
            print(f"Wrote MCD overview {path}")

    if args.only in {"all", "openmars"}:
        for path in ingest_openmars(args.source_root, args.backend_data):
            print(f"Synced OpenMARS {path}")

    if args.only in {"all", "nomad"}:
        for path in ingest_nomad(
            args.source_root,
            args.backend_data,
            args.years,
            int(args.nomad_start_my),
            args.nomad_tolerance_ls,
        ):
            print(f"Wrote NOMAD {path}")


if __name__ == "__main__":
    main()
