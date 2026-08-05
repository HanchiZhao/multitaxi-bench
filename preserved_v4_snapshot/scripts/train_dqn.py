"""Train the optional dynamic-action Double DQN and keep the best validation checkpoint."""
from __future__ import annotations

import argparse
import math
import random
from collections import deque
from copy import deepcopy

import numpy as np
import pandas as pd

from config import (
    DECISION_TIME_STEP_MINUTES, DEFAULT_COMPETITION_SCENARIO,
    DQN_BATCH_SIZE, DQN_DYNAMIC_CANDIDATES, DQN_EPISODES,
    DQN_LEARNING_RATE, DQN_REPLAY_CAPACITY, DQN_REPLAY_WARMUP,
    DQN_TARGET_UPDATE_STEPS, DQN_TRAIN_EVERY_STEPS,
    DQN_VALIDATION_EPISODES, DQN_VALIDATION_INTERVAL,
    HORIZON_MINUTES, MODELS_DIR, PROCESSED_DIR, Q_LEARNING_GAMMA,
    RANDOM_SEED, ensure_directories,
)
from dqn_policy import DQNPolicy, build_network, state_action_features
from policies import _sample_one_decision
from two_hour_environment import DynamicTaxiEnvironment, ScenarioRandomness, WAIT_ACTION, parse_clock_time


def main() -> None:
    try:
        import torch
        import torch.nn.functional as F
        import torch.optim as optim
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for DQN. Install requirements_optional_dqn.txt or run without --include-dqn."
        ) from exc

    parser = argparse.ArgumentParser(description="Train dynamic state-action Double DQN")
    parser.add_argument("--start-zone", type=int, default=132)
    parser.add_argument("--start-time", type=str, default="08:00")
    parser.add_argument("--episodes", type=int, default=DQN_EPISODES)
    parser.add_argument("--candidate-zones", type=int, default=DQN_DYNAMIC_CANDIDATES)
    parser.add_argument("--validation-episodes", type=int, default=DQN_VALIDATION_EPISODES)
    parser.add_argument("--validation-interval", type=int, default=DQN_VALIDATION_INTERVAL)
    parser.add_argument("--competition-scenario", choices=["low", "medium", "high"], default=DEFAULT_COMPETITION_SCENARIO)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    ensure_directories()

    # The DQN scores only a small dynamic action set. Limiting CPU threads avoids severe
    # overhead from large BLAS thread pools and makes training practical on ordinary laptops.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env = DynamicTaxiEnvironment.load().with_competition_scenario(args.competition_scenario)
    if args.start_zone not in env.zone_to_index:
        raise ValueError(f"Unknown start zone {args.start_zone}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = len(state_action_features(env, args.start_zone, 0.0, args.start_time, WAIT_ACTION))
    hidden_dims = (192, 128, 64)
    online = build_network(input_dim, hidden_dims).to(device)
    target = build_network(input_dim, hidden_dims).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = optim.AdamW(online.parameters(), lr=DQN_LEARNING_RATE, weight_decay=1e-5)
    replay = deque(maxlen=DQN_REPLAY_CAPACITY)
    start_abs = parse_clock_time(args.start_time)
    history: list[dict[str, float]] = []
    validation_history: list[dict[str, float]] = []
    best_validation = -float("inf")
    best_state = deepcopy(online.state_dict())
    global_step = 0

    def actions_for(zone: int, elapsed: float) -> list[str | int]:
        remaining = HORIZON_MINUTES - float(elapsed)
        dynamic = env.dynamic_candidate_zones(zone, start_abs + elapsed, n=args.candidate_zones)
        actions: list[str | int] = [WAIT_ACTION]
        for candidate in dynamic:
            if candidate != zone and env.route(zone, candidate, absolute_minutes=start_abs + elapsed)["duration_min"] < remaining:
                actions.append(int(candidate))
        return actions

    def q_values(network, zone: int, elapsed: float, actions: list[str | int]):
        feats = np.stack([
            state_action_features(env, zone, elapsed, args.start_time, action) for action in actions
        ])
        tensor = torch.tensor(feats, dtype=torch.float32, device=device)
        return network(tensor).squeeze(-1), tensor

    def next_action_features(zone: int, elapsed: float, done: bool) -> np.ndarray:
        if done or elapsed >= HORIZON_MINUTES - 1e-9:
            return np.empty((0, input_dim), dtype=np.float32)
        actions = actions_for(int(zone), float(elapsed))
        return np.stack([
            state_action_features(env, int(zone), float(elapsed), args.start_time, action)
            for action in actions
        ]).astype(np.float32)


    def evaluate_validation(episode_number: int) -> float:
        policy = DQNPolicy(online, args.start_time, args.candidate_zones, device=device)
        values = []
        for sid in range(args.validation_episodes):
            result = env.simulate_episode(
                policy, args.start_zone, args.start_time,
                scenario_id=1_000_000 + sid, base_seed=args.seed + 777,
                allowed_reposition_zones=None, horizon_minutes=HORIZON_MINUTES,
                record_events=False, dynamic_candidate_count=args.candidate_zones,
            )
            values.append(result.net_earnings)
        mean = float(np.mean(values)) if values else float("nan")
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        validation_history.append({
            "episode": int(episode_number),
            "validation_mean_net_earnings": mean,
            "validation_std": std,
            "validation_episodes": int(len(values)),
        })
        return mean

    for episode in range(1, args.episodes + 1):
        progress = (episode - 1) / max(1, args.episodes - 1)
        epsilon = 0.03 + (1.0 - 0.03) * math.exp(-5.0 * progress)
        zone, elapsed, total_reward, event_index = args.start_zone, 0.0, 0.0, 0
        scenario = ScenarioRandomness(episode - 1, args.seed)

        while elapsed < HORIZON_MINUTES - 1e-9:
            actions = actions_for(zone, elapsed)
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                with torch.no_grad():
                    q, _ = q_values(online, zone, elapsed, actions)
                action = actions[int(torch.argmax(q).item())]

            pair_feature = state_action_features(env, zone, elapsed, args.start_time, action)
            old_elapsed = elapsed
            next_zone, next_elapsed, reward, done = _sample_one_decision(
                env, zone, elapsed, start_abs, HORIZON_MINUTES, action, scenario, event_index
            )
            replay.append((
                pair_feature, float(reward),
                next_action_features(int(next_zone), float(next_elapsed), bool(done)),
                bool(done), float(next_elapsed - old_elapsed),
            ))
            total_reward += float(reward)
            zone, elapsed = int(next_zone), float(next_elapsed)
            event_index += 1
            global_step += 1

            if (
                len(replay) >= max(DQN_BATCH_SIZE, DQN_REPLAY_WARMUP)
                and global_step % DQN_TRAIN_EVERY_STEPS == 0
            ):
                batch = random.sample(replay, DQN_BATCH_SIZE)
                features = torch.tensor(np.stack([x[0] for x in batch]), dtype=torch.float32, device=device)
                rewards = torch.tensor([x[1] for x in batch], dtype=torch.float32, device=device)
                current_q = online(features).squeeze(-1)

                next_matrices = [x[2] for x in batch]
                lengths = [len(matrix) for matrix in next_matrices]
                nonempty = [matrix for matrix in next_matrices if len(matrix)]
                next_values = np.zeros(DQN_BATCH_SIZE, dtype=np.float32)
                if nonempty:
                    concatenated = np.concatenate(nonempty, axis=0)
                    tensor_next = torch.tensor(concatenated, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        online_next = online(tensor_next).squeeze(-1)
                        target_next = target(tensor_next).squeeze(-1)
                    cursor = 0
                    for index, length in enumerate(lengths):
                        if length <= 0:
                            continue
                        segment_online = online_next[cursor:cursor + length]
                        segment_target = target_next[cursor:cursor + length]
                        chosen = int(torch.argmax(segment_online).item())
                        next_values[index] = float(segment_target[chosen].item())
                        cursor += length

                discounts = np.array([
                    Q_LEARNING_GAMMA ** max(1e-6, float(x[4]) / DECISION_TIME_STEP_MINUTES)
                    for x in batch
                ], dtype=np.float32)
                targets = rewards + torch.tensor(discounts * next_values, dtype=torch.float32, device=device)
                loss = F.smooth_l1_loss(current_q, targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), 5.0)
                optimizer.step()


            if global_step % DQN_TARGET_UPDATE_STEPS == 0:
                target.load_state_dict(online.state_dict())
            if done:
                break

        if episode % max(1, args.episodes // 250) == 0 or episode == 1:
            history.append({
                "episode": episode,
                "epsilon": epsilon,
                "episode_net_reward": total_reward,
                "replay_size": len(replay),
                "global_step": global_step,
            })
        if episode % args.validation_interval == 0 or episode == args.episodes:
            validation_mean = evaluate_validation(episode)
            print(
                f"episode={episode} train_reward={total_reward:.2f} "
                f"validation={validation_mean:.2f} epsilon={epsilon:.3f}"
            )
            if validation_mean > best_validation:
                best_validation = validation_mean
                best_state = deepcopy(online.state_dict())

    payload = {
        "state_dict": {k: v.cpu() for k, v in best_state.items()},
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "start_time": args.start_time,
        "start_zone": args.start_zone,
        "candidate_count": args.candidate_zones,
        "competition_scenario": args.competition_scenario,
        "dynamic_state_action_model": True,
        "double_dqn": True,
        "best_validation_mean_net_earnings": best_validation,
        "training_episodes": args.episodes,
    }
    out = MODELS_DIR / "dqn_policy.pt"
    torch.save(payload, out)
    pd.DataFrame(history).to_csv(PROCESSED_DIR / "dqn_training_history.csv", index=False)
    pd.DataFrame(validation_history).to_csv(PROCESSED_DIR / "dqn_validation_history.csv", index=False)
    print(f"Saved best DQN model: {out}")
    print(f"Best validation mean net earnings: {best_validation:.2f}")


if __name__ == "__main__":
    main()
