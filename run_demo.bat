@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Create it first with: python -m venv venv
    exit /b 1
)

set CACHE_ONLY=true
echo Starting VeriNews in offline demo mode.
echo Open http://127.0.0.1:8000
"venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
endlocal
