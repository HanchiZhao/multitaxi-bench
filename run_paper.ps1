$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Run START_MIGRATION.ps1 first."
}

Write-Host "Verifying the locked January-June real dataset before the paper run..." -ForegroundColor Yellow
.\.venv\Scripts\python.exe scripts\verify_data.py --strict-lock
if ($LASTEXITCODE -ne 0) { throw "Real-data verification failed." }

Write-Host "Starting the full V4 + V5.1 real-data pipeline." -ForegroundColor Cyan
Write-Host "No raw data will be downloaded or overwritten." -ForegroundColor Cyan
.\.venv\Scripts\python.exe scripts\run_v51.py --config configs\paper_main.yaml --skip-download
if ($LASTEXITCODE -ne 0) { throw "Full paper pipeline failed." }
