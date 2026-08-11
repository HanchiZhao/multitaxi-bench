$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Python = if ($env:MULTITAXI_PYTHON) {
    $env:MULTITAXI_PYTHON
} else {
    Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment not found. Set MULTITAXI_PYTHON or run .\setup.ps1 first."
}

Write-Host "Verifying the locked January-June 2025 dataset..." -ForegroundColor Yellow
& $Python scripts\verify_data.py --strict-lock
if ($LASTEXITCODE -ne 0) { throw "Real-data verification failed." }

Write-Host "Starting the complete 120-minute paper pipeline." -ForegroundColor Cyan
Write-Host "No raw data will be downloaded or overwritten." -ForegroundColor Cyan
& $Python scripts\run_v52.py --config configs\paper_main.yaml --skip-download
if ($LASTEXITCODE -ne 0) { throw "Full paper pipeline failed." }

Write-Host "Running final 300-scenario acceptance checks..." -ForegroundColor Yellow
& $Python scripts\final_experiment_acceptance.py `
    --config processed_data\resolved_v52_config.yaml
if ($LASTEXITCODE -ne 0) { throw "Final experiment acceptance failed." }

Write-Host "V5.2 PAPER PIPELINE AND ACCEPTANCE PASSED" -ForegroundColor Green
