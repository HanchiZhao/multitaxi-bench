# MultiTaxi-Bench V5.1 Preservation-First

This package keeps the complete V4 Final operational pipeline and adds V5 architecture as
non-destructive extensions. The immutable original source is under `preserved_v4_snapshot/`.
The live `scripts/` directory contains every V4 entry point plus reproducibility, explicit
cost accounting, and algorithm-conditioned path-instance Shapley extensions.

## Invariants

- arbitrary start Taxi Zone and start time;
- empty driver at episode start;
- fixed 120-minute horizon;
- repeated WAIT / empty REPOSITION decisions;
- passenger destinations are exogenous;
- completed-trip revenue only;
- strict time accounting;
- complete V4 Hierarchical EB, duration calibration, latent competition, DP, Q-learning,
  Double DQN, common scenario bank, paired evaluation, global Dynamic Zone Shapley,
  imputation validation, ablation, baseline sensitivity, axioms and competition sensitivity.

## Windows

```powershell
cd C:\Users\25735\Desktop\multitaxi-bench-v5.1
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\run_smoke.ps1
powershell -ExecutionPolicy Bypass -File .\run_paper.ps1
```

See `完整替换与运行教程_CN.md` and `V4功能保留矩阵.md`.
