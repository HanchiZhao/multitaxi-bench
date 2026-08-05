# V4 输出文件清单

## 动态环境

- `processed_data/dynamic_environment.pkl`
- `processed_data/dynamic_node_metrics.csv`
- `processed_data/dynamic_od_metrics.csv`
- `processed_data/observed_od_metrics.pkl`
- `processed_data/global_od_metrics.csv`
- `processed_data/time_bin_profiles.csv`
- `processed_data/reposition_edges.csv`
- `processed_data/zone_centroids.csv`
- `processed_data/environment_build_summary.csv`

## 算法训练

- `models/q_learning_policy.pkl`
- `processed_data/q_learning_training_history.csv`
- `models/dqn_policy.pt`
- `processed_data/dqn_training_history.csv`
- `processed_data/dqn_validation_history.csv`

## 策略评估

- `processed_data/policy_episode_results.csv`
- `processed_data/algorithm_recommendation_comparison.csv`
- `processed_data/paired_policy_episode_differences.csv`
- `processed_data/paired_policy_comparisons.csv`
- `processed_data/learning_policy_diagnostics.csv`
- `processed_data/policy_trajectories.csv`
- `processed_data/representative_trajectories.csv`
- `processed_data/trajectory_event_summary.csv`
- `processed_data/policy_action_samples.csv`
- `processed_data/policy_action_agreement_matrix.csv`
- `processed_data/policy_action_type_agreement_matrix.csv`
- `processed_data/policy_action_summary.csv`

## Dynamic Zone Shapley

- `processed_data/shapley_candidate_zones.csv`
- `processed_data/node_shapley_values.csv`
- `processed_data/shapley_coalition_values.csv`
- `processed_data/shapley_run_summary.csv`
- `processed_data/shapley_zone_diagnostics.csv`
- `processed_data/shapley_value_calibration.csv`

## 验证与敏感性

- `processed_data/imputation_holdout_validation.csv`
- `processed_data/imputation_holdout_summary.csv`
- `processed_data/duration_tail_validation_summary.csv`
- `processed_data/ablation_study_results.csv`
- `processed_data/ablation_study_summary.csv`
- `processed_data/baseline_sensitivity_results.csv`
- `processed_data/baseline_sensitivity_summary.csv`
- `processed_data/axiom_validation_results.csv`
- `processed_data/axiom_validation_summary.csv`
- `processed_data/competition_sensitivity_episode_results.csv`
- `processed_data/competition_sensitivity_results.csv`
- `processed_data/competition_sensitivity_diagnostics.csv`

## 图像

- `results/figures/algorithm_net_earnings_comparison.png`
- `results/figures/paired_policy_gains_vs_wait_only.png`
- `results/figures/algorithm_operational_breakdown.png`
- `results/figures/algorithm_two_hour_trajectories.png`
- `results/figures/algorithm_two_hour_timelines.png`
- `results/figures/algorithm_completed_trip_distribution.png`
- `results/figures/top_dynamic_zone_shapley.png`
- `results/figures/dynamic_zone_shapley_map.png`
- `results/figures/shapley_ablation_results.png`
- `results/figures/imputation_holdout_validation.png`
- `results/figures/q_learning_training_curve.png`
- `results/figures/dqn_training_curve.png`
- `results/figures/policy_action_agreement_matrix.png`
- `results/figures/policy_action_type_agreement_matrix.png`
- `results/figures/policy_wait_action_rates.png`
- `results/figures/competition_sensitivity.png`
- `results/figures/demand_supply_wait_maps.png`
