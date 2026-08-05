# V5.1 Preservation-First 完整迁移教程

本文件已被更新。请以项目根目录中的：

`FULL_MIGRATION_GUIDE_CN.md`

作为当前唯一完整教程。

当前状态：

- 2025-01 至 2025-06 六个月真实数据全部已包含；
- 正式运行不再自动下载或覆盖原始数据；
- Smoke Test 已隔离，不会覆盖真实数据；
- V4 Preservation Check 和 Research Contract Check 继续保留。

最短流程：

```powershell
powershell -ExecutionPolicy Bypass -File .\START_MIGRATION.ps1
powershell -ExecutionPolicy Bypass -File .\run_smoke.ps1
powershell -ExecutionPolicy Bypass -File .\run_paper.ps1
```
