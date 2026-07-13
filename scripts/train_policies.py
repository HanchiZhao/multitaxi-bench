"""Train tabular Q-learning with dynamic state-specific reposition actions."""
from __future__ import annotations

import argparse
import pandas as pd

from config import POLICY_CANDIDATE_ZONES, PROCESSED_DIR, Q_LEARNING_EPISODES, RANDOM_SEED, ensure_directories
from policies import train_q_learning
from two_hour_environment import DynamicTaxiEnvironment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dynamic-action Q-learning policy")
    parser.add_argument("--start-zone", type=int, default=132)
    parser.add_argument("--start-time", type=str, default="08:00")
    parser.add_argument("--episodes", type=int, default=Q_LEARNING_EPISODES)
    parser.add_argument("--candidate-zones", type=int, default=POLICY_CANDIDATE_ZONES,
                        help="Only used to export the initial-state diagnostic candidate list")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    ensure_directories()
    env = DynamicTaxiEnvironment.load()
    initial = env.dynamic_candidate_zones(args.start_zone, args.start_time, n=args.candidate_zones)
    policy, history = train_q_learning(
        env, args.start_zone, args.start_time, candidate_zones=None,
        episodes=args.episodes, seed=args.seed,
    )
    policy.save()
    pd.DataFrame(history).to_csv(PROCESSED_DIR / "q_learning_training_history.csv", index=False)
    pd.DataFrame({
        "rank": range(1, len(initial)+1),
        "zone_id": initial,
        "zone_name": [env.zone_names.get(z,str(z)) for z in initial],
    }).to_csv(PROCESSED_DIR / "policy_candidate_zones.csv", index=False)
    print(f"Q-learning trained for {args.episodes:,} episodes with dynamic actions.")
    print(f"  {PROCESSED_DIR / 'q_learning_training_history.csv'}")


if __name__ == "__main__":
    main()
