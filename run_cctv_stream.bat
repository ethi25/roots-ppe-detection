@echo off
title Roots Industrial PPE Compliance Detector - Live External CCTV Stream
cd /d "%~dp0"
echo ========================================================================
echo       ROOTS INDUSTRIAL PPE COMPLIANCE DETECTOR - CCTV NVR ACCESS
echo ========================================================================
echo.
echo Make sure you are connected to the factory CCTV Wi-Fi network!
echo (Press ENTER to accept the default values in brackets [ ])
echo.

set CCTV_IP=192.168.127.5
set /p CCTV_IP="Enter CCTV NVR IP Address [default: 192.168.127.5]: "

set CCTV_CH=29
set /p CCTV_CH="Enter Camera Channel [default: 29 for D29]: "

set CCTV_USER=admin
set /p CCTV_USER="Enter Username [default: admin]: "

set CCTV_PASS=dmin@123
set /p CCTV_PASS="Enter Password [default: dmin@123]: "

set CCTV_AREA=POWDER COATING OFFICE AREA
set /p CCTV_AREA="Enter Area Name [default: POWDER COATING OFFICE AREA]: "

echo.
echo [*] Connecting to NVR %CCTV_IP% on Camera Channel D%CCTV_CH% (%CCTV_AREA%)...
echo.

python terminal_detector.py --ip "%CCTV_IP%" --channel "%CCTV_CH%" --user "%CCTV_USER%" --password "%CCTV_PASS%" --brand "hikvision" --area "%CCTV_AREA%"

pause
