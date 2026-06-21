r"""
Clean generated project outputs for the multitaxi-bench pipeline.

This script deletes only generated artifacts from:
- processed_data/
- results/figures/

It never deletes raw files in data/.

Run from project root:
    python scripts/clean_outputs.py

The environment script also calls this cleaner automatically when you run:
    python scripts/route_environment.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed_data"
FIGURES_DIR = BASE_DIR / "results" / "figures"

# processed_data is treated as generated output in this project.
# Keep only placeholder files such as .gitkeep if present.
PROCESSED_PATTERNS = [
    "*.csv",
    "*.pkl",
    "*.json",
    "*.txt",
]

# Generated figures. Keep placeholder files such as .gitkeep if present.
FIGURE_PATTERNS = [
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.pdf",
    "*.svg",
]

KEEP_FILENAMES = {
    ".gitkeep",
    ".gitignore",
}


def _delete_matching_files(directory: Path, patterns: Iterable[str], dry_run: bool = False) -> List[Path]:
    deleted: List[Path] = []
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        return deleted

    for pattern in patterns:
        for path in directory.glob(pattern):
            if not path.is_file():
                continue
            if path.name in KEEP_FILENAMES:
                continue
            deleted.append(path)
            if not dry_run:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
    return deleted


def clean_generated_outputs(dry_run: bool = False, verbose: bool = True) -> List[Path]:
    """Delete generated processed-data and figure outputs.

    This keeps the raw data/ directory untouched. It is safe to run before a full
    pipeline rebuild. The route_environment.py script calls this function by
    default at startup so old outputs do not mix with new outputs.
    """
    deleted: List[Path] = []
    deleted.extend(_delete_matching_files(PROCESSED_DIR, PROCESSED_PATTERNS, dry_run=dry_run))
    deleted.extend(_delete_matching_files(FIGURES_DIR, FIGURE_PATTERNS, dry_run=dry_run))

    if verbose:
        action = "Would delete" if dry_run else "Deleted"
        if deleted:
            print(f"[clean_outputs] {action} {len(deleted)} generated file(s):")
            for p in deleted:
                try:
                    print(f"  - {p.relative_to(BASE_DIR)}")
                except ValueError:
                    print(f"  - {p}")
        else:
            print("[clean_outputs] No generated output files to clean.")
    return deleted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean generated multitaxi-bench outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting files.")
    parser.add_argument("--quiet", action="store_true", help="Do not print deleted file names.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_generated_outputs(dry_run=args.dry_run, verbose=not args.quiet)


if __name__ == "__main__":
    main()
