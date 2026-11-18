param(
    [string]$TripsDir = "data/student_test",
    [ValidateSet("nearest", "classifier", "hmm")][string]$Mode = "nearest",
    [ValidateRange(1, 65535)][int]$Port = 8000
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) { throw "Create .venv and install requirements.txt first. See README.md." }
$env:PATHSNAP_TRIPS_DIR = $TripsDir
$env:PATHSNAP_MODE = $Mode
Write-Host "PathSnap | mode=$Mode | trips=$TripsDir"
Write-Host "Laptop 2: open http://<laptop-1-Wi-Fi-IPv4>:$Port/ui/"
Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -ne "127.0.0.1" -and $_.IPAddress -notlike "169.254*" } |
    Format-Table InterfaceAlias,IPAddress
& $pythonExe -m uvicorn api.main:app --host 0.0.0.0 --port $Port
exit $LASTEXITCODE
