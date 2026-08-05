$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
powershell -ExecutionPolicy Bypass -File .\START_MIGRATION.ps1
