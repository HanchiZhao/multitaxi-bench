# MultiTaxi-Bench V5.1 新电脑完整迁移教程（2025-01 至 2025-06）

## 0. 这次迁移包是什么

这是 **V5.1 Preservation-First** 的完整迁移版：

- V4 Final 的 23 个核心脚本仍然保留；
- V4 的 Hierarchical EB、120 分钟动态环境、Common Scenario Bank、Value Iteration、Q-learning、Double DQN、V4 Dynamic Zone Shapley、公理检验、消融、插补验证、竞争敏感性等仍由 V4 实现；
- V5.1 只在其上增加现实成本模型、算法—路径—节点实例 Path Shapley、数据锁、复现脚本和 GitHub 结构；
- 本包已经包含你提供的 **2025 年 1—6 月全部真实 Yellow Taxi parquet** 和 Taxi Zone 文件；
- 正式运行时不会再自动下载或覆盖这些原始数据。

## 1. 已核对的真实数据

当前包中存在：

- `data\yellow_tripdata_2025-01.parquet`
- `data\yellow_tripdata_2025-02.parquet`
- `data\yellow_tripdata_2025-03.parquet`
- `data\yellow_tripdata_2025-04.parquet`
- `data\yellow_tripdata_2025-05.parquet`
- `data\yellow_tripdata_2025-06.parquet`
- `data\taxi_zone_lookup.csv`
- `data\taxi_zones.shp`
- `data\taxi_zones.shx`
- `data\taxi_zones.dbf`
- `data\taxi_zones.prj`
- `data\taxi_zones.cpg`

静态检查结果：

- 6 个 parquet 的文件头和文件尾均为 `PAR1`；
- 6 个月都能在 parquet 元数据中找到项目需要的字段名，包括：
  `tpep_pickup_datetime`、`tpep_dropoff_datetime`、`PULocationID`、`DOLocationID`、`trip_distance`、`fare_amount`、`tip_amount`、`cbd_congestion_fee`；
- Taxi Zone lookup：265 个 LocationID；其中 264 = Unknown，265 = Outside of NYC；
- Taxi Zone shapefile：263 个实际空间区域，263/263 geometry 有效；
- CRS：`EPSG:2263`。

安装环境后，`START_MIGRATION.ps1` 还会使用 PyArrow 对 **全部 6 个月**重新做 schema 验证，并使用 `data_lock.json` 对所有数据逐文件做 SHA256 校验。

## 2. 最推荐的解压位置

为了避免 Windows 路径、OneDrive 同步、中文路径或权限问题，推荐解压到一个短路径：

```text
C:\MultiTaxiBenchV51
```

不要覆盖旧的 V4 项目。如果你旧电脑上的项目已经不在了，也没有关系，本包包含恢复项目所需的源码结构。

## 3. 需要提前安装什么

需要：

- Windows 10/11 64-bit
- **Python 3.11 x64**
- 建议至少 16 GB RAM
- 建议至少 15–20 GB 可用磁盘空间（原始数据本身约 391 MB，但环境、模型和输出会继续占空间）

Git 只在你之后要上传 GitHub 时需要；运行实验本身不依赖 Git。

## 4. 第一步：打开 PowerShell

最简单的方法：

1. 用文件资源管理器打开 `C:\MultiTaxiBenchV51`；
2. 点击窗口顶部地址栏；
3. 输入：

```text
powershell
```

4. 按 Enter。

这样 PowerShell 会直接打开在正确目录，不需要手动写用户名路径。

## 5. 第二步：一键完成迁移检查

运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\START_MIGRATION.ps1
```

它会依次：

1. 检查 1—6 月真实数据和完整 Taxi Zone 文件是否存在；
2. 创建 `.venv`；
3. 安装固定版本依赖；
4. 安装 CPU 版 PyTorch，保证新电脑无 GPU 也能复现；
5. 对 6 个 parquet 做真实 schema 检查；
6. 对所有迁移数据做 SHA256 校验；
7. 检查 V4 核心源码是否完整保留；
8. 检查 120 分钟研究目标是否被修改。

正常末尾应看到：

```text
DATA VERIFICATION PASSED
V4 PRESERVATION PASSED
RESEARCH CONTRACT PASSED
MIGRATION READY
```

如果任何一项不通过，不要继续跑正式实验，把完整终端截图发给我。

## 6. 第三步：运行安全 Smoke Test

运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_smoke.ps1
```

### 这版 Smoke Test 和以前的关键区别

**它绝不会覆盖你真实的 `data` 文件。**

以前的 `generate_mock_data.py --force` 如果直接在项目根目录运行，会覆盖真实 parquet。现在 `run_smoke.ps1` 会先创建：

```text
_smoke_workspace\
```

所有 mock 数据、processed data、模型和结果都只写在这个隔离目录。

真实文件：

```text
data\yellow_tripdata_2025-01.parquet
...
data\yellow_tripdata_2025-06.parquet
```

不会被 Smoke 修改。

正常末尾应出现：

```text
SAFE SMOKE TEST PASSED
```

Smoke 输出位于：

```text
_smoke_workspace\processed_data
_smoke_workspace\results
```

## 7. 第四步：正式运行六个月真实数据

Smoke 通过以后执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_paper.ps1
```

正式脚本会先再次执行：

```text
verify_data.py --strict-lock
```

确认数据没被改动，然后运行完整：

```text
V4 Preservation Check
→ Research Contract Check
→ Hierarchical EB Environment
→ Full OD Coverage
→ Common Scenario Bank
→ Q-learning
→ Double DQN
→ All Policy Evaluation
→ Paired Comparisons
→ V4 Dynamic Zone Shapley
→ V4 Axiom Validation
→ Imputation Validation
→ Ablation Study
→ Baseline Sensitivity
→ Competition Sensitivity
→ V4 Figures
→ V5.1 Cost Re-accounting
→ Algorithm-conditioned Path Shapley
→ Path-level Axiom Validation
→ Path-Shapley Figures
→ Reproduction Verification
```

### 默认正式配置

`configs\paper_main.yaml`：

- start zone = 132（JFK）
- start time = 08:00
- horizon = 120 min（由 V4 contract 固定）
- evaluation scenarios = 300
- Q-learning = 15,000 episodes
- DQN = 12,000 episodes
- primary cost model = owner-operator
- occupied cost = 0.70 USD/mile
- empty cost = 0.70 USD/mile
- Path Shapley permutations = 512

这会比 Smoke 慢很多。根据 CPU 和内存，完整任务可能需要较长时间。

## 8. 主要输出在哪里

### V4 原始输出

```text
processed_data\
results\figures\
models\
```

关键结果包括：

```text
policy_episode_results.csv
algorithm_recommendation_comparison.csv
paired_policy_comparisons.csv
learning_policy_diagnostics.csv
node_shapley_values.csv
shapley_zone_diagnostics.csv
shapley_value_calibration.csv
imputation validation outputs
ablation outputs
baseline sensitivity outputs
competition sensitivity outputs
```

### V5.1 新增输出

```text
policy_episode_cost_models.csv
algorithm_cost_model_comparison.csv
paired_cost_model_comparisons.csv
path_shapley_values.csv
path_coalition_values.csv
path_shapley_efficiency_checks.csv
path_value_calibration.csv
same_zone_cross_path_context.csv
path_axiom_validation_results.csv
```

新增图像包括：

```text
policy_take_home_comparison.png
cost_model_comparison.png
algorithm_path_shapley_bars.png
algorithm_conditioned_path_shapley_maps.png
cross_algorithm_zone_shapley_heatmap.png
```

## 9. 随时检查数据有没有损坏

运行：

```powershell
.\.venv\Scripts\python.exe scripts\verify_data.py --strict-lock
```

正常结果：

```text
DATA VERIFICATION PASSED
```

## 10. 随时检查 V4 有没有被改坏

运行：

```powershell
.\.venv\Scripts\python.exe scripts\verify_v4_preservation.py
```

正常结果：

```text
V4 PRESERVATION PASSED
```

## 11. 随时检查研究目标有没有被暗改

运行：

```powershell
.\.venv\Scripts\python.exe scripts\verify_research_contract.py --static-only
```

它检查的核心约束包括：

- 120 分钟固定时域；
- 司机空车开始；
- 动作仍是 WAIT / REPOSITION；
- 乘客目的地仍然外生；
- 只有时域内完成订单计入完成收入；
- unfinished occupied 和 terminal unused time 仍存在；
- 时间核算仍受容差约束。

## 12. GitHub 迁移建议

原始 parquet 和大型生成文件不要提交 GitHub。`.gitignore` 已排除这些内容。

建议先建立新分支：

```powershell
git init
git remote add origin https://github.com/HanchiZhao/multitaxi-bench.git
git fetch origin
git checkout -b v5.1-preservation-first-migrated

git add .
git status
git commit -m "Restore V5.1 preservation-first project with reproducible Jan-Jun 2025 workflow"
git push -u origin v5.1-preservation-first-migrated
```

在 `git status` 中确认不要出现：

```text
data\yellow_tripdata_*.parquet
processed_data\*.pkl
models\*.pt
```

## 13. 如果大压缩包仍然打不开

这次我同时提供：

1. 一个小型 **CODE** 压缩包；
2. `Jan–Mar` 数据包；
3. `Apr–Jun` 数据包；
4. Taxi Zone 数据包。

如果完整大包下载/解压有问题，使用这四个小包即可。

操作方法：

1. 解压 CODE 包到 `C:\MultiTaxiBenchV51`；
2. 把三个数据包都“解压到 `C:\MultiTaxiBenchV51`”；
3. 它们内部已经带 `data\` 目录，因此最终应看到：

```text
C:\MultiTaxiBenchV51\data\yellow_tripdata_2025-01.parquet
...
C:\MultiTaxiBenchV51\data\yellow_tripdata_2025-06.parquet
C:\MultiTaxiBenchV51\data\taxi_zones.shp
```

4. 然后运行 `START_MIGRATION.ps1`。

## 14. 最重要的禁忌

不要手动运行：

```powershell
python scripts\generate_mock_data.py --force
```

对正式项目来说这条命令可能覆盖真实数据。

需要 Smoke 时只运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_smoke.ps1
```

新版 Smoke 已经隔离。

---

如果 `START_MIGRATION.ps1`、`run_smoke.ps1` 或 `run_paper.ps1` 中任何一步报错，保留终端从命令开始到最后错误行的完整截图，我可以按当前迁移版继续定位，不需要重新描述项目背景。
