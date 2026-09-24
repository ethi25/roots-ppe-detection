@echo off
title Roots PPE Detector - Live High-Speed Webcam
cd /d "%~dp0"
python terminal_detector.py --video 0
pause
