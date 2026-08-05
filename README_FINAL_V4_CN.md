# MultiTaxi-Bench 动态两小时最终版 V4

## 最终研究设定

给定任意 Taxi Zone 起点和任意开始时间，司机以空车状态开始。在未来 120 分钟内，策略反复选择：

- `WAIT`：留在当前区域等待乘客；
- `REPOSITION_TO(zone)`：空车移动到另一个区域。

乘客目的地由历史/Hierarchical EB OD 分布决定。所有策略统一以两小时累计运营净收益评价。

## V4 相比 V3.1 的主要修正

1. 两小时所有分钟完整核算：等待、空驶、已完成载客、跨界未完成载客、终止未使用时间之和严格等于 120 分钟。
2. 输出同一 scenario 下相对 `wait_only` 的配对收益差、配对 95% CI 和胜率。
3. 行程时长改为 log-scale EB，并加入时间、距离分段、borough pair、机场 OD 的收缩校准，减少长行程被过度压缩。
4. Q-learning 输出 Q-table 决策率、fallback 率、未见状态率。
5. DQN 升级为动态 state-action Double DQN，使用经验回放、Huber loss、目标网络和固定验证集，并保存验证集最佳模型。
6. 候选区域生成不再每次扫描全纽约所有 Zone；使用本地邻区、时间段城市级候选、乘客高概率目的地和历史 OD 联合池，保持多样性的同时显著提升训练速度。
7. Shapley 新增玩家数据置信度诊断，以及 DP 期望值与同场景仿真结果校准。
8. 竞争敏感性加入 95% CI，并可同时评估固定 Q-learning 和 DQN。
9. 轨迹图增加方向和编号；新增完整事件时间线与完成订单数分布，避免重叠路线造成“订单很少”的错觉。

## 替换方法

只需用压缩包内的 `scripts` 整体替换：

```text
C:\Users\25735\Desktop\multitaxi-bench\scripts
```

建议先将旧文件夹改名为：

```text
scripts_v3_1_backup
```

不要删除 `data`。

因为 V4 改变了环境统计模型和 pickle schema，第一次运行必须重新构建环境，不要加 `--skip-environment`。

## 安装依赖

```bat
cd C:\Users\25735\Desktop\multitaxi-bench
python -m pip install -r requirements_dynamic.txt
python -m pip install -r requirements_optional_dqn.txt
```

## 第一次快速测试，包括 Q-learning 与 DQN

```bat
cd C:\Users\25735\Desktop\multitaxi-bench && python scripts\run_all.py --start-zone 132 --start-time 08:00 --fast --include-q-learning --include-dqn
```

## 正式完整运行，包括 Q-learning 与 DQN

```bat
cd C:\Users\25735\Desktop\multitaxi-bench && python scripts\run_all.py --start-zone 132 --start-time 08:00 --include-q-learning --include-dqn
```

默认正式训练：

- Q-learning：10,000 episodes；
- Double DQN：8,000 episodes；
- 策略评估：300 个共同场景；
- Shapley：10 个玩家，自动使用 exact Shapley；
- 竞争敏感性：低、中、高三个情景。

可手动调整训练量：

```bat
python scripts\run_all.py --start-zone 132 --start-time 08:00 --include-q-learning --include-dqn --q-episodes 15000 --dqn-episodes 12000
```

## 环境已由 V4 成功构建后，跳过环境重建

```bat
python scripts\run_all.py --start-zone 132 --start-time 08:00 --include-q-learning --include-dqn --skip-environment
```

## 单独查询任意时间和任意 OD

```bat
python scripts\query_od.py --origin 132 --destination 161 --time 08:07
```

输出预期收入、时长、距离、标准差、置信度、历史样本数和估值来源。

## 重要解释

地图上的多条订单可能使用相同 Taxi Zone 起终点，因此地理线会重叠。V4 新增：

- `algorithm_two_hour_timelines.png`：按时间显示每一次等待、载客和空驶；
- `algorithm_completed_trip_distribution.png`：显示所有场景中的完成订单数分布；
- `trajectory_event_summary.csv`：逐策略统计事件数量和总时长。

这些文件比单独看地图更适合判断两小时内到底完成了多少单。
