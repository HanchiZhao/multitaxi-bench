"""Fast self-checks for the final V4 dynamic pipeline."""
from __future__ import annotations

import math

from policies import WaitOnlyPolicy
from shapley_explainer import CoalitionValueCache, exact_shapley
from two_hour_environment import DynamicTaxiEnvironment, FiniteHorizonModel


def main() -> None:
    env=DynamicTaxiEnvironment.load(); start=env.zone_ids[0]; time="08:07"
    candidates=env.dynamic_candidate_zones(start,time,n=min(3,len(env.zone_ids)-1))
    if not candidates: raise AssertionError("No dynamic candidate zones")
    # Every valid OD query must be finite, including a pair absent from the time-specific support.
    est=env.estimate_od(start,env.zone_ids[-1],time)
    for k in ["expected_driver_revenue","expected_duration_minutes","expected_distance_miles","confidence"]:
        if not math.isfinite(float(est[k])): raise AssertionError(f"Non-finite arbitrary OD estimate: {k}")
    model=FiniteHorizonModel(env,time,120,candidates)
    empty=model.start_value(start,[]); full=model.start_value(start,candidates)
    if full+1e-9<empty: raise AssertionError("Action-set monotonicity failed")
    cache=CoalitionValueCache(model,start,candidates); phi,_=exact_shapley(cache)
    gap=float(phi.sum()-(cache.value((1<<len(candidates))-1)-cache.value(0)))
    if abs(gap)>1e-7: raise AssertionError(f"Shapley efficiency failed: {gap}")
    result=env.simulate_episode(WaitOnlyPolicy(),start,time,0,123,None)
    if not math.isfinite(result.net_earnings): raise AssertionError("Simulation returned non-finite earnings")
    if abs(result.accounted_minutes - 120.0) > 1e-7:
        raise AssertionError(f"Two-hour minute accounting failed: {result.accounted_minutes}")
    if abs(result.time_accounting_error) > 1e-7:
        raise AssertionError(f"Non-zero time accounting error: {result.time_accounting_error}")
    waits=[env.with_competition_scenario(s).node_metric(start,480)["expected_wait_min"] for s in ["low","medium","high"]]
    if not (waits[0]<=waits[1]<=waits[2]): raise AssertionError("Competition scenario wait ordering failed")
    print("SELF-CHECK PASSED")
    print(f"start_zone={start}; candidates={candidates}")
    print(f"arbitrary_od_source={est['estimate_source']}; confidence={float(est['confidence']):.3f}")
    print(f"wait_low_medium_high={waits}")
    print(f"wait_only_expected_value={empty:.4f}; full_action_expected_value={full:.4f}")


if __name__=="__main__": main()
