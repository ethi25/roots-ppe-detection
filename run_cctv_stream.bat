@echo off
title Roots Industrial PPE Compliance Detector - Live External CCTV Stream
cd /d "%~dp0"
echo ========================================================================
echo       ROOTS INDUSTRIAL PPE COMPLIANCE DETECTOR - CCTV NVR ACCESS
echo ========================================================================
echo.
echo Make sure you are connected to the factory CCTV Wi-Fi network!
echo.
set /p CCTV_IP="Enter CCTV NVR IP Address (e.g. 192.168.1.100): "
if "%CCTV_IP%"=="" (
    echo Starting interactive menu...
    python terminal_detector.py
    goto end
)

set /p CCTV_CH="Enter Camera Channel [e.g. 29 for D29] (default: 29): "
if "%CCTV_CH%"=="" set CCTV_CH=29

set /p CCTV_USER="Enter Username (default: admin): "
if "%CCTV_USER%"=="" set CCTV_USER=admin

set /p CCTV_PASS="Enter Password (press Enter if none): "

echo.
echo [*] Connecting to NVR %CCTV_IP% on Camera Channel D%CCTV_CH%...
echo.

python terminal_detector.py --ip "%CCTV_IP%" --channel "%CCTV_CH%" --user "%CCTV_USER%" --password "%CCTV_PASS%"

:end
pause
