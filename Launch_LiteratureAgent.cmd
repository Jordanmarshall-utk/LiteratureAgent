@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo LiteratureAgent is not set up yet.
  echo Run Setup_LiteratureAgent.cmd first.
  pause
  exit /b 1
)

set "PYTHONPATH=%CD%"
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8501/_stcore/health' -TimeoutSec 2 ^| Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
  start "LiteratureAgent Server" /min ".venv\Scripts\python.exe" -m streamlit run "literature_agent_platform\app.py" --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
  timeout /t 4 /nobreak >nul
)
start "LiteratureAgent" "http://127.0.0.1:8501"
