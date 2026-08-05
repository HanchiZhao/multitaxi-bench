# 新电脑迁移说明（已更新为完整 1—6 月数据版）

当前迁移包已经包含 2025 年 1—6 月全部 Yellow Taxi parquet 和完整 Taxi Zone 文件。

请不要再按旧说明下载 6 月数据。直接打开：

- `START_HERE.txt`：最短操作说明；
- `FULL_MIGRATION_GUIDE_CN.md`：完整教程。

第一条命令：

```powershell
powershell -ExecutionPolicy Bypass -File .\START_MIGRATION.ps1
```

之后先运行安全 Smoke：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_smoke.ps1
```

再运行正式六个月真实数据：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_paper.ps1
```

新版 `run_smoke.ps1` 在 `_smoke_workspace` 中生成 mock 数据，不会覆盖 `data` 中的真实 parquet。
