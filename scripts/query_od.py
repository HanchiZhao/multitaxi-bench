"""Query one arbitrary taxi-zone OD estimate at any clock time."""
from __future__ import annotations

import argparse
import json
from two_hour_environment import DynamicTaxiEnvironment


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate distance, duration and revenue for any OD-time pair")
    parser.add_argument("--origin", type=int, required=True)
    parser.add_argument("--destination", type=int, required=True)
    parser.add_argument("--time", type=str, required=True, help="HH:MM, e.g. 08:07")
    args = parser.parse_args()
    env = DynamicTaxiEnvironment.load()
    result = env.estimate_od(args.origin, args.destination, args.time)
    result["origin_name"] = env.zone_names.get(args.origin, str(args.origin))
    result["destination_name"] = env.zone_names.get(args.destination, str(args.destination))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
