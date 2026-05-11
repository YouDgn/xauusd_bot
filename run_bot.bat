@echo off
REM ============================================================
REM XAUUSD AI Trading Bot Launcher
REM ============================================================
REM Double-click pour lancer le bot automatiquement

setlocal enabledelayedexpansion

cd /d "%~dp0"

REM Vérifier si venv existe
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo ============================================================
    echo ERREUR: Virtual Environment non trouve !
    echo ============================================================
    echo.
    echo Crée d'abord le venv avec :
    echo   python -m venv venv
    echo   venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Activer venv et lancer le bot
echo.
echo ============================================================
echo  XAUUSD AI Trading Bot - Starting...
echo ============================================================
echo.

call venv\Scripts\activate.bat
python live_bot.py

REM Garder la fenêtre ouverte en cas d'erreur
if errorlevel 1 (
    echo.
    echo ============================================================
    echo ERREUR: Le bot s'est arrete
    echo ============================================================
    echo.
    pause
)
