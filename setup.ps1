$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function New-ProjectVenv {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 -c "import sys; print(sys.version)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Using Python 3.11 through the Windows py launcher." -ForegroundColor Cyan
            & py -3.11 -m venv .venv
            if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv with Python 3.11." }
            return
        }
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        $ver = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($ver.Trim() -eq "3.11") {
            Write-Host "Using python.exe version 3.11." -ForegroundColor Cyan
            & python -m venv .venv
            if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv with Python 3.11." }
            return
        }
    }

    throw @"
Python 3.11 x64 was not found.
Install Python 3.11 for Windows, make sure the Python launcher or python.exe is available,
then rerun .\setup.ps1.
"@
}

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    New-ProjectVenv
}

$Python = ".\.venv\Scripts\python.exe"
Write-Host "Installing pinned project dependencies..." -ForegroundColor Yellow
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
& $Python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Core dependency installation failed." }
& $Python -m pip install -r requirements-dqn.txt
if ($LASTEXITCODE -ne 0) { throw "PyTorch CPU installation failed." }

Write-Host "Checking the audited V4 core manifest..." -ForegroundColor Yellow
& $Python scripts\verify_v4_preservation.py
if ($LASTEXITCODE -ne 0) { throw "V4 preservation check failed." }

Write-Host "Running preflight..." -ForegroundColor Yellow
& $Python scripts\preflight.py --config configs\smoke.yaml
if ($LASTEXITCODE -ne 0) { throw "Preflight failed." }

Write-Host "SETUP PASSED" -ForegroundColor Green
