$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Run START_MIGRATION.ps1 first."
}

# IMPORTANT: never generate mock data in the real data/ directory.
# The entire smoke run is isolated in a disposable workspace.
$SmokeRoot = Join-Path $PSScriptRoot "_smoke_workspace"
if (Test-Path $SmokeRoot) {
    Remove-Item -Recurse -Force $SmokeRoot
}
New-Item -ItemType Directory -Path $SmokeRoot | Out-Null

Write-Host "Creating isolated smoke workspace..." -ForegroundColor Yellow
Copy-Item -Recurse -Force ".\scripts" (Join-Path $SmokeRoot "scripts")
Copy-Item -Recurse -Force ".\configs" (Join-Path $SmokeRoot "configs")
Copy-Item -Recurse -Force ".\preserved_v4_snapshot" (Join-Path $SmokeRoot "preserved_v4_snapshot")

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Push-Location $SmokeRoot
try {
    Write-Host "Generating mock data ONLY inside _smoke_workspace..." -ForegroundColor Yellow
    & $Python scripts\generate_mock_data.py --trips-per-month 1200 --force
    if ($LASTEXITCODE -ne 0) { throw "Mock-data generation failed." }

    Write-Host "Running the preservation-first smoke pipeline..." -ForegroundColor Yellow
    & $Python scripts\run_v51.py --config configs\smoke.yaml --skip-download
    if ($LASTEXITCODE -ne 0) { throw "Smoke pipeline failed." }
}
finally {
    Pop-Location
}

Write-Host "" 
Write-Host "SAFE SMOKE TEST PASSED" -ForegroundColor Green
Write-Host "Real files under .\data were never touched." -ForegroundColor Green
Write-Host "Smoke outputs are under .\_smoke_workspace\results and .\_smoke_workspace\processed_data." -ForegroundColor Cyan
