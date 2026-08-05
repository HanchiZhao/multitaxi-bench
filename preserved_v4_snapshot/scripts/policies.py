"""Operating policies for the two-hour taxi environment."""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from config import (
    DECISION_TIME_STEP_MINUTES,
    EMPTY_COST_PER_MILE,
    HORIZON_MINUTES,
    MODELS_DIR,
    OCCUPIED_COST_PER_MILE,
    POLICY_CANDIDATE_ZONES,
    Q_LEARNING_ALPHA,
    Q_LEARNING_EPISODES,
    Q_LEARNING_EPSILON_END,
    Q_LEARNING_EPSILON_START,
    Q_LEARNING_GAMMA,
    RANDOM_SEED,
    TIME_BIN_MINUTES,
)
from two_hour_environment import (
    DynamicProgramPolicy,
    DynamicTaxiEnvironment,
    FiniteHorizonModel,
    ScenarioRandomness,
    TaxiState,
    WAIT_ACTION,
    parse_clock_time,
)


class BasePolicy:
    name = "base"
    def select_action(self, env: DynamicTaxiEnvironment, state: TaxiState,
                      start_absolute_minutes: float, allowed_zones: Sequence[int]) -> str | int:
        raise NotImplementedError


class WaitOnlyPolicy(BasePolicy):
    name = "wait_only"
    def select_action(self, env, state, start_absolute_minutes, allowed_zones):
        return WAIT_ACTION


class HighestDemandPolicy(BasePolicy):
    name = "highest_demand"
    def select_action(self, env, state, start_absolute_minutes, allowed_zones):
        absolute = start_absolute_minutes + state.elapsed_minutes
        current = env.node_metric(state.zone_id, absolute)
        best_zone = state.zone_id
        best_score = current["pickup_rate_index"] / max(1e-6, current["competition_ratio"])
        for z in allowed_zones:
            route = env.route(state.zone_id, int(z), absolute_minutes=absolute)
            if route["duration_min"] >= state.remaining_minutes: continue
            metric = env.node_metric(int(z), absolute + route["duration_min"])
            # Demand policy is intentionally simple, but it now considers competition:
            # demand divided by latent queue pressure rather than raw pickup count alone.
            score = metric["pickup_rate_index"] / max(1e-6, metric["competition_ratio"])
            if score > best_score + 1e-12:
                best_zone, best_score = int(z), score
        return WAIT_ACTION if best_zone == state.zone_id else best_zone


class HighestIncomePolicy(BasePolicy):
    name = "highest_income"
    def select_action(self, env, state, start_absolute_minutes, allowed_zones):
        absolute = start_absolute_minutes + state.elapsed_minutes
        current = env.node_metric(state.zone_id, absolute)
        best_zone = state.zone_id
        best_score = current["expected_trip_revenue"] - OCCUPIED_COST_PER_MILE*current["expected_trip_distance_miles"]
        for z in allowed_zones:
            route = env.route(state.zone_id, int(z), absolute_minutes=absolute)
            if route["duration_min"] >= state.remaining_minutes: continue
            metric = env.node_metric(int(z), absolute + route["duration_min"])
            score = metric["expected_trip_revenue"] - OCCUPIED_COST_PER_MILE*metric["expected_trip_distance_miles"]
            if score > best_score + 1e-12:
                best_zone, best_score = int(z), score
        return WAIT_ACTION if best_zone == state.zone_id else best_zone


class GreedyNetEarningsPolicy(BasePolicy):
    name = "greedy_net_earnings"

    @staticmethod
    def cycle_rate(env: DynamicTaxiEnvironment, zone: int, absolute: float) -> Tuple[float,float]:
        m = env.node_metric(zone, absolute)
        net = m["expected_trip_revenue"] - OCCUPIED_COST_PER_MILE*m["expected_trip_distance_miles"]
        cycle = max(1.0, m["expected_wait_min"] + m["expected_trip_duration_min"])
        return net/cycle, net

    def select_action(self, env, state, start_absolute_minutes, allowed_zones):
        absolute = start_absolute_minutes + state.elapsed_minutes
        best_rate, _ = self.cycle_rate(env, state.zone_id, absolute)
        best_zone = state.zone_id
        for z in allowed_zones:
            z = int(z)
            if z == state.zone_id: continue
            route = env.route(state.zone_id, z, absolute_minutes=absolute)
            reposition_time = float(route["duration_min"])
            if reposition_time >= state.remaining_minutes: continue
            metric = env.node_metric(z, absolute + reposition_time)
            future_net = metric["expected_trip_revenue"] - OCCUPIED_COST_PER_MILE*metric["expected_trip_distance_miles"]
            denominator = max(1.0, reposition_time + metric["expected_wait_min"] + metric["expected_trip_duration_min"])
            score = (future_net - EMPTY_COST_PER_MILE*float(route["distance_miles"]))/denominator
            if score > best_rate + 1e-12:
                best_rate, best_zone = score, z
        return WAIT_ACTION if best_zone == state.zone_id else best_zone


class LegacyUtilityPolicy(BasePolicy):
    name = "legacy_utility_greedy"
    def select_action(self, env, state, start_absolute_minutes, allowed_zones):
        absolute = start_absolute_minutes + state.elapsed_minutes
        current = env.node_metric(state.zone_id, absolute)
        best_zone = state.zone_id
        best = current["expected_trip_revenue"] + 0.3*current["pickup_rate_index"]
        for z in allowed_zones:
            route = env.route(state.zone_id, int(z), absolute_minutes=absolute)
            if route["duration_min"] >= state.remaining_minutes: continue
            m = env.node_metric(int(z), absolute+route["duration_min"])
            score = m["expected_trip_revenue"] + 0.3*m["pickup_rate_index"] - 0.5*route["duration_min"] - 0.15*route["distance_miles"]
            if score > best: best, best_zone = score, int(z)
        return WAIT_ACTION if best_zone == state.zone_id else best_zone


@dataclass
class QLearningPolicy(BasePolicy):
    """Tabular Q policy with dynamic actions and a greedy fallback for unseen states."""
    q_table: Dict[Tuple[int,int,int], Dict[str,float]]
    start_time: str
    horizon_minutes: int = HORIZON_MINUTES
    name: str = "q_learning"
    decision_count: int = 0
    q_table_action_count: int = 0
    fallback_action_count: int = 0
    unseen_state_count: int = 0
    no_valid_seen_action_count: int = 0

    def reset_diagnostics(self) -> None:
        self.decision_count = 0
        self.q_table_action_count = 0
        self.fallback_action_count = 0
        self.unseen_state_count = 0
        self.no_valid_seen_action_count = 0

    def diagnostics(self) -> Dict[str, float]:
        total = max(1, self.decision_count)
        return {
            "algorithm": self.name,
            "decision_count": int(self.decision_count),
            "q_table_state_count": int(len(self.q_table)),
            "q_table_action_count": int(self.q_table_action_count),
            "fallback_action_count": int(self.fallback_action_count),
            "unseen_state_count": int(self.unseen_state_count),
            "no_valid_seen_action_count": int(self.no_valid_seen_action_count),
            "q_table_action_rate": float(self.q_table_action_count / total),
            "fallback_action_rate": float(self.fallback_action_count / total),
            "unseen_state_rate": float(self.unseen_state_count / total),
        }

    def state_key(self, zone_id: int, elapsed_minutes: float) -> Tuple[int,int,int]:
        absolute = parse_clock_time(self.start_time) + elapsed_minutes
        time_bin = int((absolute%1440)//TIME_BIN_MINUTES)
        remaining_bucket = max(0, int(math.ceil((self.horizon_minutes-elapsed_minutes)/DECISION_TIME_STEP_MINUTES)))
        return int(zone_id), time_bin, remaining_bucket

    def select_action(self, env, state, start_absolute_minutes, allowed_zones):
        self.decision_count += 1
        key = self.state_key(state.zone_id, state.elapsed_minutes)
        q = self.q_table.get(key)
        valid = [WAIT_ACTION] + [str(int(z)) for z in allowed_zones if int(z)!=state.zone_id]
        if q is not None:
            seen = [a for a in valid if a in q]
            if seen:
                best_value = max(float(q[a]) for a in seen)
                # Deterministic but WAIT-neutral tie break: prefer the lowest-cost action only
                # after comparing Q values, rather than insertion order.
                tied = [a for a in seen if abs(float(q[a]) - best_value) <= 1e-12]
                best_key = sorted(tied, key=lambda a: (a != WAIT_ACTION, str(a)))[0]
                self.q_table_action_count += 1
                return WAIT_ACTION if best_key == WAIT_ACTION else int(best_key)
            self.no_valid_seen_action_count += 1
        else:
            self.unseen_state_count += 1
        self.fallback_action_count += 1
        # A sensible fallback avoids the previous unseen-state -> WAIT collapse.
        return GreedyNetEarningsPolicy().select_action(env,state,start_absolute_minutes,allowed_zones)

    def save(self, path: Path | str | None = None) -> Path:
        path = Path(path or (MODELS_DIR/"q_learning_policy.pkl")); path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("wb") as f: pickle.dump(self,f,pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path: Path | str | None = None) -> "QLearningPolicy":
        with Path(path or (MODELS_DIR/"q_learning_policy.pkl")).open("rb") as f: return pickle.load(f)


def _sample_one_decision(env: DynamicTaxiEnvironment, zone: int, elapsed: float, start_abs: float,
                         horizon_minutes: int, action: str|int, scenario: ScenarioRandomness,
                         event_index: int) -> Tuple[int,float,float,bool]:
    remaining=horizon_minutes-elapsed; absolute=start_abs+elapsed
    if action == WAIT_ACTION:
        m=env.node_metric(zone,absolute); mean=max(0.1,m["expected_wait_min"])
        wait=-mean*math.log(max(1e-12,scenario.uniform(event_index,"q_wait")))
        if wait > DECISION_TIME_STEP_MINUTES:
            dt=min(float(DECISION_TIME_STEP_MINUTES),remaining)
            return zone,elapsed+dt,0.0,elapsed+dt>=horizon_minutes
        options=env.od_options(zone,absolute+wait)
        if not options: return zone,min(horizon_minutes,elapsed+wait),0.0,elapsed+wait>=horizon_minutes
        u=scenario.uniform(event_index,"q_dest"); cum=0.0; selected=options[-1]
        for o in options:
            cum+=o["probability"]
            if u<=cum: selected=o; break
        revenue,duration,distance=env.sample_trip_metrics(selected,scenario,event_index)
        end=elapsed+wait+duration
        if end>horizon_minutes: return zone,float(horizon_minutes),0.0,True
        return int(selected["destination"]),end,revenue-OCCUPIED_COST_PER_MILE*distance,False
    target=int(action); route=env.route(zone,target,absolute_minutes=absolute); duration=float(route["duration_min"])
    if duration>=remaining: return zone,float(horizon_minutes),0.0,True
    return target,elapsed+duration,-EMPTY_COST_PER_MILE*float(route["distance_miles"]),False


def train_q_learning(env: DynamicTaxiEnvironment, start_zone: int, start_time: str,
                     candidate_zones: Optional[Sequence[int]]=None,
                     episodes: int=Q_LEARNING_EPISODES, seed: int=RANDOM_SEED) -> Tuple[QLearningPolicy,List[Dict[str,float]]]:
    """Train with state-specific dynamic candidate actions.

    candidate_zones remains accepted for backward compatibility; when provided it is used as
    an optional global ceiling, but state-specific candidates are still recomputed.
    """
    global_ceiling = None if candidate_zones is None else set(map(int,candidate_zones))
    q_table: Dict[Tuple[int,int,int],Dict[str,float]]={}; rng=np.random.default_rng(seed)
    start_abs=parse_clock_time(start_time); history=[]

    def key(zone:int,elapsed:float)->Tuple[int,int,int]:
        return int(zone),int(((start_abs+elapsed)%1440)//TIME_BIN_MINUTES),max(0,int(math.ceil((HORIZON_MINUTES-elapsed)/DECISION_TIME_STEP_MINUTES)))

    for episode in range(int(episodes)):
        frac=episode/max(1,episodes-1); epsilon=Q_LEARNING_EPSILON_START+frac*(Q_LEARNING_EPSILON_END-Q_LEARNING_EPSILON_START)
        zone=int(start_zone); elapsed=0.0; total=0.0; event_index=0; scenario=ScenarioRandomness(episode,seed)
        while elapsed<HORIZON_MINUTES-1e-9:
            state_key=key(zone,elapsed); q=q_table.setdefault(state_key,{WAIT_ACTION:0.0})
            dynamic=env.dynamic_candidate_zones(zone,start_abs+elapsed,n=POLICY_CANDIDATE_ZONES)
            if global_ceiling is not None:
                # Keep dynamic alternatives first, then allow ceiling zones to avoid a frozen initial action set.
                dynamic=list(dict.fromkeys(dynamic+[z for z in global_ceiling if z!=zone]))
            actions=[WAIT_ACTION]+[str(z) for z in dynamic if env.route(zone,z,absolute_minutes=start_abs+elapsed)["duration_min"]<HORIZON_MINUTES-elapsed]
            for a in actions: q.setdefault(a,0.0)
            if rng.random() < epsilon:
                action_key = str(rng.choice(actions))
            else:
                best_q = max(q[a] for a in actions)
                tied = [a for a in actions if abs(q[a] - best_q) <= 1e-12]
                # Random tie-breaking avoids the historical insertion-order bias toward WAIT.
                action_key = str(rng.choice(tied))
            action=WAIT_ACTION if action_key==WAIT_ACTION else int(action_key)
            old_elapsed = elapsed
            nz,ne,reward,done=_sample_one_decision(env,zone,elapsed,start_abs,HORIZON_MINUTES,action,scenario,event_index)
            next_max=0.0
            if not done:
                nk=key(nz,ne); nq=q_table.setdefault(nk,{WAIT_ACTION:0.0})
                next_dynamic=env.dynamic_candidate_zones(nz,start_abs+ne,n=POLICY_CANDIDATE_ZONES)
                next_actions=[WAIT_ACTION]+[str(z) for z in next_dynamic if env.route(nz,z,absolute_minutes=start_abs+ne)["duration_min"]<HORIZON_MINUTES-ne]
                for a in next_actions: nq.setdefault(a,0.0)
                next_max=max(nq[a] for a in next_actions)
            duration_steps = max(1e-6, (ne - old_elapsed) / DECISION_TIME_STEP_MINUTES)
            semi_markov_discount = Q_LEARNING_GAMMA ** duration_steps
            q[action_key] += Q_LEARNING_ALPHA * (reward + semi_markov_discount * next_max - q[action_key])
            total+=reward; zone,elapsed=nz,ne; event_index+=1
            if done: break
        if episode%max(1,episodes//200)==0 or episode==episodes-1:
            history.append({"episode":episode+1,"epsilon":epsilon,"episode_net_reward":total,"q_state_count":len(q_table)})
    return QLearningPolicy(q_table=q_table,start_time=str(start_time)),history


def make_core_policies(env: DynamicTaxiEnvironment, start_zone: int, start_time: str,
                       candidate_zones: Optional[Sequence[int]]=None, include_legacy: bool=False,
                       q_policy: Optional[QLearningPolicy]=None) -> List[BasePolicy]:
    # candidate_zones=None enables the model's state-specific dynamic action generator.
    model=FiniteHorizonModel(env,start_time,HORIZON_MINUTES,candidate_zones=None)
    policies: List[BasePolicy]=[WaitOnlyPolicy(),HighestDemandPolicy(),HighestIncomePolicy(),GreedyNetEarningsPolicy(),DynamicProgramPolicy(model,None)]
    if q_policy is not None: policies.append(q_policy)
    if include_legacy: policies.append(LegacyUtilityPolicy())
    return policies
