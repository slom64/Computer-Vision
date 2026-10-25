@echo off
title Sensum PATVIS APA - Fluid Bed Agglomeration & Sticking Monitor
cd /d "%~dp0"

echo =========================================================================
echo    SENSUM PATVIS APA - IN-LINE FLUID BED PARTICLE INSPECTION SYSTEM
echo =========================================================================
echo.
echo Camera Target: Daheng Imaging MER2-301-125U3M (USB 3.0, 3.1MP Monochrome)
echo Initializing environment...
echo.

set PYTHON_EXE=D:\Programs\miniforge\envs\deeplearning\python.exe

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python environment not found at: %PYTHON_EXE%
    echo Falling back to system python...
    set PYTHON_EXE=python
)

"%PYTHON_EXE%" run_patvis.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%.
    pause
)
