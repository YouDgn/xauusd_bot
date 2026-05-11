# ============================================================
# XAUUSD AI Trading Bot Launcher (PowerShell)
# ============================================================
# Exécute avec : .\run_bot.ps1

param(
    [switch]$NoActivate = $false
)

# Récupérer le répertoire courant
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host ""
Write-Host "============================================================"
Write-Host "  XAUUSD AI Trading Bot - Starting..."
Write-Host "============================================================"
Write-Host ""

# Vérifier venv
$venvPath = ".\venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvPath)) {
    Write-Host "ERREUR: Virtual Environment non trouve !" -ForegroundColor Red
    Write-Host ""
    Write-Host "Crée d'abord le venv avec :"
    Write-Host "  python -m venv venv"
    Write-Host "  venv\Scripts\pip install -r requirements.txt"
    Write-Host ""
    Read-Host "Appuie sur Enter pour fermer"
    exit 1
}

# Activer venv
& $venvPath

# Lancer le bot
python live_bot.py

# Garder la fenêtre ouverte en cas d'erreur
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "ERREUR: Le bot s'est arrete (Code: $LASTEXITCODE)" -ForegroundColor Red
    Write-Host "============================================================"
    Write-Host ""
    Read-Host "Appuie sur Enter pour fermer"
}
