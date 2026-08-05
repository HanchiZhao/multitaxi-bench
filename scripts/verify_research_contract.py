"""Fail-fast research-contract checks preventing silent target changes."""
from __future__ import annotations
import argparse, inspect, json
from pathlib import Path
from config import HORIZON_MINUTES, DECISION_TIME_STEP_MINUTES, WAIT_REEVALUATION_MINUTES, PROCESSED_DIR, DRIVER_REVENUE_SHARE, OCCUPIED_COST_PER_MILE, EMPTY_COST_PER_MILE
from two_hour_environment import DynamicTaxiEnvironment, WAIT_ACTION
from policies import WaitOnlyPolicy
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--static-only",action="store_true"); args=ap.parse_args()
    checks={
      "horizon_is_120":HORIZON_MINUTES==120,
      "decision_step_is_5":DECISION_TIME_STEP_MINUTES==5,
      "wait_revaluation_is_5":WAIT_REEVALUATION_MINUTES==5,
      "wait_action_exists":WAIT_ACTION=="WAIT",
      "positive_revenue_share":0<DRIVER_REVENUE_SHARE<=1,
      "nonnegative_mileage_costs":OCCUPIED_COST_PER_MILE>=0 and EMPTY_COST_PER_MILE>=0,
    }
    src=inspect.getsource(DynamicTaxiEnvironment.simulate_episode)
    checks["passenger_destination_exogenous"]="self.od_options" in src and 'policy.select_action' in src
    checks["completed_trip_revenue_only"]="gross += revenue" in src and "UNFINISHED_TRIP" in src
    checks["strict_time_accounting_present"]="accounting_error" in src and "unfinished_occupied" in src and "terminal_unused" in src
    if not args.static_only:
        env=DynamicTaxiEnvironment.load(); zone=int(env.zone_ids[0]); result=env.simulate_episode(WaitOnlyPolicy(),zone,"08:00",999999,20260712,record_events=True)
        checks["driver_starts_at_requested_zone"]=result.start_zone==zone
        checks["episode_horizon_is_120"]=result.horizon_minutes==120
        checks["time_accounting_runtime"]=abs(result.time_accounting_error)<=1e-6
        checks["actions_are_wait_or_reposition"]=all((str(e.selected_action)==WAIT_ACTION) or str(e.selected_action).isdigit() or str(e.selected_action)=="END" for e in result.events)
    failed=[k for k,v in checks.items() if not v]
    report={"status":"RESEARCH CONTRACT PASSED" if not failed else "RESEARCH CONTRACT FAILED","checks":checks,"failed":failed,"active_objective":{"driver_revenue_share":DRIVER_REVENUE_SHARE,"occupied_cost_per_mile":OCCUPIED_COST_PER_MILE,"empty_cost_per_mile":EMPTY_COST_PER_MILE}}
    PROCESSED_DIR.mkdir(exist_ok=True); (PROCESSED_DIR/"research_contract_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    if failed: raise SystemExit(1)
if __name__=="__main__": main()
