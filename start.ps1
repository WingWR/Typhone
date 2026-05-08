$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSCommandPath
$python    = "python.exe"
$backend   = Join-Path $projectRoot "backend\app.py"
$frontend  = Join-Path $projectRoot "frontend"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " TyphoonAI - starting backend + frontend" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# ── Backend ──────────────────────────────────────────
Write-Host "`n[1/2] Starting Flask backend on port 5000 ..." -ForegroundColor Yellow
$backendJob = Start-Process -FilePath $python -ArgumentList $backend -PassThru -WindowStyle Normal

Start-Sleep -Seconds 3

try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/health" -TimeoutSec 5
    if ($health.status -eq "ok") {
        Write-Host "       Backend ready: http://127.0.0.1:5000" -ForegroundColor Green
    }
} catch {
    Write-Host "       Backend may still be starting, check http://127.0.0.1:5000/api/health" -ForegroundColor DarkYellow
}

# ── Frontend ─────────────────────────────────────────
Write-Host "`n[2/2] Starting Vite frontend on port 5173 ..." -ForegroundColor Yellow
$frontendJob = Start-Process -FilePath "cmd" -ArgumentList "/c","npm run dev" -WorkingDirectory $frontend -PassThru -WindowStyle Normal

Write-Host "       Frontend: http://127.0.0.1:5173" -ForegroundColor Green

# ── Done ─────────────────────────────────────────────
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " Both services started." -ForegroundColor Cyan
Write-Host " Backend  - http://127.0.0.1:5000" -ForegroundColor White
Write-Host " Frontend - http://127.0.0.1:5173" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nPress Enter to stop both services..." -ForegroundColor DarkGray
Read-Host | Out-Null

Stop-Process -Id $backendJob.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $frontendJob.Id -Force -ErrorAction SilentlyContinue
Write-Host "Done." -ForegroundColor Gray
