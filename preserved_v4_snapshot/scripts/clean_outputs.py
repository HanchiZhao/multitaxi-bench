"""Remove generated outputs without touching raw data."""
from __future__ import annotations

import argparse
from pathlib import Path

from config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, ensure_directories


def collect_targets(include_models: bool = False):
    patterns = ["*.csv", "*.pkl", "*.json", "*.txt"]
    targets = []
    for pattern in patterns:
        targets.extend(PROCESSED_DIR.glob(pattern))
    for pattern in ["*.png", "*.jpg", "*.jpeg", "*.pdf", "*.svg"]:
        targets.extend(FIGURES_DIR.glob(pattern))
    if include_models:
        for pattern in ["*.pkl", "*.pt", "*.pth", "*.json"]:
            targets.extend(MODELS_DIR.glob(pattern))
    return sorted(set(targets))


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean generated MultiTaxi outputs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-models", action="store_true")
    args = parser.parse_args()
    ensure_directories()
    targets = collect_targets(args.include_models)
    for path in targets:
        print(("Would remove" if args.dry_run else "Removing") + f": {path}")
        if not args.dry_run:
            path.unlink(missing_ok=True)
    print(f"{len(targets)} generated files {'listed' if args.dry_run else 'removed'}.")


if __name__ == "__main__":
    main()
