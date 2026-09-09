@echo off
title NeuroBreathe AI - Dual Clinical Diagnostics
echo =========================================================
echo       Starting NeuroBreathe AI Platform...
echo =========================================================

REM Check for virtual environment python
if exist "neurobreathe-ai\ml_training\venv\Scripts\python.exe" (
    echo [Environment] Found dedicated project venv.
    "neurobreathe-ai\ml_training\venv\Scripts\python.exe" run_app.py
) else if exist "%USERPROFILE%\neurobreathe-ai\ml_training\venv\Scripts\python.exe" (
    echo [Environment] Found user profile venv.
    "%USERPROFILE%\neurobreathe-ai\ml_training\venv\Scripts\python.exe" run_app.py
) else (
    echo [Environment] Using system python.
    python run_app.py
)

pause
