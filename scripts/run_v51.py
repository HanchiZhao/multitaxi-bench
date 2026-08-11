"""Compatibility entry point redirected to the mandatory v5.2 pipeline.

The former v5.1 runner is intentionally unavailable because it did not require the
181-service-day date-boundary validation. Existing automation may keep this filename,
but execution always enters the v5.2 runner and its gates.
"""
from __future__ import annotations

from run_v52 import main


if __name__ == "__main__":
    print(
        "run_v51.py is superseded; redirecting to the mandatory v5.2 "
        "date-boundary-validated pipeline."
    )
    main()
