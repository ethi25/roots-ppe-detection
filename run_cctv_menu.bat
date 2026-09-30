@echo off
title Roots Industrial PPE Compliance Detector - Factory CCTV Camera Selector
cd /d "%~dp0"

:menu
cls
echo ========================================================================
echo       ROOTS INDUSTRIAL PPE COMPLIANCE DETECTOR - FACTORY CCTV SELECTOR
echo ========================================================================
echo.
echo Select Camera Location to Monitor:
echo.
echo   [1] [D3]  PC SPRAY BOOTH WEST WALL
echo   [2] [D23] TOOL ROOM
echo   [3] [D29] POWDER COATING OFFICE AREA
echo   [4] Enter Custom Channel or IP
echo   [5] Exit
echo.

set CHOICE=2
set /p CHOICE="Enter Choice [1-4, default: 2]: "
if "%CHOICE%"=="1" goto d3
if "%CHOICE%"=="2" goto d23
if "%CHOICE%"=="3" goto d29
if "%CHOICE%"=="4" goto custom
if "%CHOICE%"=="5" exit
goto d23

:d3
call run_cctv_d3.bat
goto menu

:d23
call run_cctv_d23.bat
goto menu

:d29
call run_cctv_d29.bat
goto menu

:custom
call run_cctv_stream.bat
goto menu
