@echo off
title Roots Industrial PPE Compliance Detector - [D3] PC SPRAY BOOTH WEST WALL
cd /d "%~dp0"
echo ========================================================================
echo   ROOTS INDUSTRIAL PPE COMPLIANCE DETECTOR - [D3] PC SPRAY BOOTH WEST WALL
echo ========================================================================
echo.
echo Connecting to Live CCTV Stream:
echo   - Camera Location : [D3] PC SPRAY BOOTH WEST WALL
echo   - NVR IP Address  : 192.168.127.5
echo   - Camera Channel  : Digital Channel 3 (Sub-Stream Ch 302)
echo   - RTSP Port       : 554 (TCP Low-Latency Transport)
echo   - Multi-Worker AI : ByteTrack Tracking Active
echo   - Critical Gear   : Respirator / Mask (Cyan) + Safety Gloves (Yellow)
echo.
echo Press 'Q' inside video window to quit, 'P' to pause/resume.
echo.

python terminal_detector.py --ip "192.168.127.5" --channel "3" --user "admin" --password "dmin@123" --brand "hikvision" --area "PC SPRAY BOOTH WEST WALL"

pause
