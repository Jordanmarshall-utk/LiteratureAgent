@echo off
setlocal
cd /d "%~dp0"
echo This optional setup downloads the larger 7B text and vision models.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_and_launch_platform.ps1" -SetupOnly -ModelProfile quality
if errorlevel 1 pause
exit /b %errorlevel%
