@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_and_launch_platform.ps1" -SetupOnly -ModelProfile laptop
if errorlevel 1 pause
exit /b %errorlevel%
