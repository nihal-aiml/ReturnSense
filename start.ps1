# ReturnSense — Windows Dev Launcher
# ====================================
# Starts both the frontend dev server and FastAPI backend.
#
# Usage: .\start.ps1

Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║        📦 ReturnSense Dev Environment        ║" -ForegroundColor Cyan
Write-Host "  ╠══════════════════════════════════════════════╣" -ForegroundColor Cyan
Write-Host "  ║  Frontend:  http://localhost:5173             ║" -ForegroundColor Cyan
Write-Host "  ║  API:       http://localhost:8000             ║" -ForegroundColor Cyan
Write-Host "  ║  API Docs:  http://localhost:8000/docs        ║" -ForegroundColor Cyan
Write-Host "  ╚══════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Start FastAPI backend in background
$apiJob = Start-Job -ScriptBlock {
    Set-Location $using:PSScriptRoot
    uvicorn api.app:app --reload --port 8000 --host 0.0.0.0
}

# Start frontend server in foreground
try {
    node server.js
}
finally {
    Write-Host "`n  Shutting down ReturnSense..." -ForegroundColor Yellow
    Stop-Job $apiJob -ErrorAction SilentlyContinue
    Remove-Job $apiJob -ErrorAction SilentlyContinue
}
