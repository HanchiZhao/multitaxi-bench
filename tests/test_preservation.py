from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REQUIRED={'route_environment.py','two_hour_environment.py','policies.py','train_policies.py','dqn_policy.py','train_dqn.py','evaluate_recommendations.py','shapley_explainer.py','imputation_validation.py','ablation_study.py','baseline_sensitivity.py','axiom_validation.py','competition_sensitivity.py','visualize_results.py','run_all.py'}
def test_v4_snapshot_complete(): assert REQUIRED <= {p.name for p in (ROOT/'preserved_v4_snapshot'/'scripts').glob('*.py')}
def test_live_v4_entrypoints_complete(): assert REQUIRED <= {p.name for p in (ROOT/'scripts').glob('*.py')}
