# TradingAgents nightly runner — called by Windows Task Scheduler
# Runs at 5:30 AM SGT (Tue-Sat), analyzing the previous US trading day's close.

$ProjectDir = Split-Path -Parent $PSScriptRoot
$PythonExe  = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$Script     = Join-Path $ProjectDir "scripts\daily.py"
$LogDir     = Join-Path $ProjectDir "logs"
$Date       = (Get-Date).AddDays(-1).ToString("yyyy-MM-dd")
$LogFile    = Join-Path $LogDir "daily_$Date.log"

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting daily run for $Date" | Tee-Object -FilePath $LogFile

Set-Location $ProjectDir

# Force UTF-8 so Rich's Unicode characters (✓, →, etc.) don't crash on cp1252
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

& $PythonExe $Script --run --skip-sync --date $Date *>&1 | Tee-Object -FilePath $LogFile -Append

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Run complete." | Tee-Object -FilePath $LogFile -Append
