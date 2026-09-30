@echo off
title Roots Industrial PPE Compliance Detector - Live External CCTV Stream
cd /d "%~dp0"
echo ========================================================================
echo       ROOTS INDUSTRIAL PPE COMPLIANCE DETECTOR - CCTV ACCESS
echo ========================================================================
echo.
echo You can connect by entering either:
echo   1. Just the Camera IP Address (e.g., 192.168.1.64)
echo   2. Or the full RTSP URL (e.g., rtsp://admin:pass@192.168.1.64:554/...)
echo.
set /p CCTV_INPUT="Enter CCTV IP or RTSP URL: "
if "%CCTV_INPUT%"=="" (
    echo Starting interactive menu...
    python terminal_detector.py
) else (
    python terminal_detector.py --video "%CCTV_INPUT%"
)
pause
