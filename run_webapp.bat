@echo off
title ROOTS PPE Compliance Detector - Web App
cd /d "%~dp0"

echo ========================================================================
echo   ROOTS INDUSTRIAL PPE COMPLIANCE DETECTOR - WEB APPLICATION
echo ========================================================================
echo.
echo Starting Backend & Frontend Services:
echo   - Backend API & AI Engine : http://127.0.0.1:8000
echo   - Web User Interface      : http://127.0.0.1:5173
echo   - Hardware Acceleration   : NVIDIA RTX 4050 GPU
echo   - Detection Architecture  : Fine-tuned YOLOv8m + ByteTrack Multi-Worker
echo.

:: 1. Launch Backend API in background window
start "ROOTS PPE - Backend Server" cmd /k "python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000 --http h11 --ws none"

:: Wait 3 seconds for backend initialization
timeout /t 3 /nobreak >nul

:: 2. Launch Frontend in background window
start "ROOTS PPE - Frontend Server" cmd /k "cd frontend && npm run dev -- --host 127.0.0.1 --port 5173"

:: Wait 2 seconds for Vite initialization
timeout /t 2 /nobreak >nul

:: 3. Launch Default Web Browser
echo Opening Web Application in Browser...
start http://127.0.0.1:5173/

echo.
echo Web application is running. Close the terminal windows to stop the servers.
echo.
pause
