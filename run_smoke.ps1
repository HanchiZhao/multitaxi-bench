$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Python = if ($env:MULTITAXI_PYTHON) {
    $env:MULTITAXI_PYTHON
} else {
    Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Run .\setup.ps1 first."
}

# Mock data and outputs are isolated from the real data directory.
$SmokeRoot = Join-Path $PSScriptRoot "_smoke_workspace"
if (Test-Path $SmokeRoot) {
    Remove-Item -Recurse -Force $SmokeRoot
}
New-Item -ItemType Directory -Path $SmokeRoot | Out-Null
Copy-Item -Recurse -Force ".\scripts" (Join-Path $SmokeRoot "scripts")
Copy-Item -Recurse -Force ".\configs" (Join-Path $SmokeRoot "configs")
Copy-Item -Recurse -Force ".\provenance" (Join-Path $SmokeRoot "provenance")

Push-Location $SmokeRoot
try {
    Write-Host "Generating mock data inside the isolated smoke workspace..." -ForegroundColor Yellow
    & $Python scripts\generate_mock_data.py --trips-per-month 1200 --force
    if ($LASTEXITCODE -ne 0) { throw "Mock-data generation failed." }

    Write-Host "Running the complete V5.2 smoke pipeline..." -ForegroundColor Yellow
    & $Python scripts\run_v52.py --config configs\smoke.yaml --skip-download
    if ($LASTEXITCODE -ne 0) { throw "Smoke pipeline failed." }
}
finally {
    Pop-Location
}

Write-Host "SAFE SMOKE TEST PASSED" -ForegroundColor Green
Write-Host "Real files under .\data were never touched." -ForegroundColor Green
Write-Host "Smoke outputs: .\_smoke_workspace\processed_data and .\_smoke_workspace\results" -ForegroundColor Cyan
