@echo off
setlocal
cd /d "%~dp0"
title Hydroponic Medium Optimizer

cls
echo ============================================================
echo   Hydroponic Medium Optimizer
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo This computer needs Python installed first.
    echo Go to https://www.python.org/downloads/
    echo ^(tick "Add Python to PATH" during installation^),
    echo then double-click this file again.
    echo.
    pause
    exit /b 1
)

set NEED_SETUP=0
if not exist ".venv\Scripts\python.exe" (
    set NEED_SETUP=1
) else (
    ".venv\Scripts\python.exe" -c "import streamlit" >nul 2>nul
    if errorlevel 1 set NEED_SETUP=1
)

if "%NEED_SETUP%"=="1" (
    echo Setting up for the first time -- please wait,
    echo this can take a few minutes. You will only see this once.
    echo.
    if exist setup_log.txt del setup_log.txt
    python -m venv .venv >> setup_log.txt 2>&1
    ".venv\Scripts\python.exe" -m pip install --upgrade pip >> setup_log.txt 2>&1
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt >> setup_log.txt 2>&1
    if errorlevel 1 (
        echo.
        echo Something went wrong during setup.
        echo Please send the file "setup_log.txt" from this folder
        echo so it can be looked at.
        echo.
        pause
        exit /b 1
    )
    echo Setup complete.
    echo.
)

echo Starting the app -- your browser will open automatically
echo in a few seconds.
echo.
echo IMPORTANT: keep this window open while you use the app.
echo When you are done, simply close this window.
echo ============================================================
echo.

".venv\Scripts\python.exe" -m streamlit run app.py > app_log.txt 2>&1

echo.
echo The app has stopped.
pause
