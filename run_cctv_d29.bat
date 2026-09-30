@echo off
title Roots Industrial PPE Compliance Detector - [D29] POWDER COATING OFFICE AREA
cd /d "%~dp0"
echo ========================================================================
echo   ROOTS INDUSTRIAL PPE COMPLIANCE DETECTOR - [D29] POWDER COATING AREA
echo ========================================================================
echo.
echo Connecting to Live CCTV Stream:
echo   - Camera Location : [D29] POWDER COATING OFFICE AREA
echo   - NVR IP Address  : 192.168.127.5
echo   - Camera Channel  : Digital Channel 29 (Sub-Stream Ch 2902)
echo   - RTSP Port       : 554 (TCP Transport)
echo   - Multi-Worker AI : ByteTrack Tracking Active
echo   - Compliance Gear : Respirator/Mask (Cyan) + Safety Gloves (Yellow)
echo.
echo Press 'Q' inside video preview to quit, 'P' to pause/resume.
echo.

python terminal_detector.py --ip "192.168.127.5" --channel "29" --user "admin" --password "dmin@123" --brand "hikvision" --area "POWDER COATING OFFICE AREA"

pause
