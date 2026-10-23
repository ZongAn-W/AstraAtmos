"""Check that every relative markdown link in the docs resolves to a real file.

Run from the project root with the project interpreter:

    & (Join-Path $env:CONDA_PREFIX 'python.exe') scripts/audit/check-doc-links.py

Only relative file links are checked; anchors, http(s) URLs and mailto are skipped.
Exits non-zero when a link points at a missing path.
"""

from __future__ import annotations

import pathlib
import re
import sys

LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
SKIP_PREFIXES = ("http://", "https://", "#", "mailto:")


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[2]
    targets = [root / "README.md", *(root / "docs").rglob("*.md")]
    missing = []
    checked = 0
    for path in targets:
        text = path.read_text(encoding="utf-8")
        for label, link in LINK.findall(text):
            if link.startswith(SKIP_PREFIXES):
                continue
            target = link.split("#", 1)[0]
            if not target:
                continue
            checked += 1
            if not (path.parent / target).resolve().exists():
                missing.append(f"{path.relative_to(root)} :: [{label}]({link})")
    print(f"checked {checked} relative links across {len(targets)} files")
    if missing:
        print("\n".join(f"MISSING {row}" for row in missing))
        return 1
    print("all relative doc links resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
