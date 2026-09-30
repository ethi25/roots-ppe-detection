@echo off
title Roots Industrial PPE Compliance Detector - [D23] TOOL ROOM
cd /d "%~dp0"
echo ========================================================================
echo       ROOTS INDUSTRIAL PPE COMPLIANCE DETECTOR - [D23] TOOL ROOM
echo ========================================================================
echo.
echo Connecting to Live CCTV Stream:
echo   - Camera Location : [D23] TOOL ROOM
echo   - NVR IP Address  : 192.168.127.5
echo   - Camera Channel  : Digital Channel 23 (Sub-Stream Ch 2302)
echo   - RTSP Port       : 554 (TCP Low-Latency Transport)
echo   - Multi-Worker AI : ByteTrack Tracking Active
echo   - Safety Gear     : Respirator / Mask (Cyan) + Safety Gloves (Yellow)
echo.
echo Press 'Q' inside video window to quit, 'P' to pause/resume.
echo.

python terminal_detector.py --ip "192.168.127.5" --channel "23" --user "admin" --password "dmin@123" --brand "hikvision" --area "TOOL ROOM"

pause
