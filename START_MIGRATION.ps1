$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " MultiTaxi-Bench V5.1 Complete Migration - Jan-Jun 2025" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$required = @(
  "data\yellow_tripdata_2025-01.parquet",
  "data\yellow_tripdata_2025-02.parquet",
  "data\yellow_tripdata_2025-03.parquet",
  "data\yellow_tripdata_2025-04.parquet",
  "data\yellow_tripdata_2025-05.parquet",
  "data\yellow_tripdata_2025-06.parquet",
  "data\taxi_zone_lookup.csv",
  "data\taxi_zones.shp",
  "data\taxi_zones.shx",
  "data\taxi_zones.dbf",
  "data\taxi_zones.prj",
  "data\taxi_zones.cpg"
)

$missing = @()
foreach ($f in $required) {
  if (-not (Test-Path $f)) { $missing += $f }
}
if ($missing.Count -gt 0) {
  Write-Host "The migration package is incomplete. Missing files:" -ForegroundColor Red
  $missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
  throw "Restore the missing files before continuing."
}

Write-Host "[1/5] Creating the pinned Python 3.11 environment..." -ForegroundColor Yellow
powershell -ExecutionPolicy Bypass -File .\setup.ps1

Write-Host "[2/5] Auditing all six real TLC months and taxi-zone files..." -ForegroundColor Yellow
.\.venv\Scripts\python.exe scripts\migration_data_audit.py

Write-Host "[3/5] Verifying exact SHA256 data lock and parquet schemas..." -ForegroundColor Yellow
.\.venv\Scripts\python.exe scripts\verify_data.py --strict-lock

Write-Host "[4/5] Confirming V4 source preservation..." -ForegroundColor Yellow
.\.venv\Scripts\python.exe scripts\verify_v4_preservation.py

Write-Host "[5/5] Confirming the research target contract..." -ForegroundColor Yellow
.\.venv\Scripts\python.exe scripts\verify_research_contract.py --static-only

Write-Host "" 
Write-Host "MIGRATION READY" -ForegroundColor Green
Write-Host "Your real January-June data have NOT been modified." -ForegroundColor Green
Write-Host "Next run the SAFE smoke test:" -ForegroundColor Cyan
Write-Host "  powershell -ExecutionPolicy Bypass -File .\run_smoke.ps1"
Write-Host "Then run the full real-data paper pipeline:" -ForegroundColor Cyan
Write-Host "  powershell -ExecutionPolicy Bypass -File .\run_paper.ps1"
