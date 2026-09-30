@echo off
title Roots Industrial PPE Compliance Detector - Live External CCTV Stream
cd /d "%~dp0"
echo ========================================================================
echo       ROOTS INDUSTRIAL PPE COMPLIANCE DETECTOR - CCTV STREAM
echo ========================================================================
echo.
echo Enter your CCTV Camera / NVR RTSP URL:
echo (Example: rtsp://admin:password@192.168.1.64:554/Streaming/Channels/102)
echo.
set /p CCTV_URL="CCTV RTSP URL: "
if "%CCTV_URL%"=="" (
    echo No URL entered. Opening interactive menu...
    python terminal_detector.py
) else (
    python terminal_detector.py --video "%CCTV_URL%"
)
pause
