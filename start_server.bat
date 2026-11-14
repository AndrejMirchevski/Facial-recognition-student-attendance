@echo off
REM Start the attendance web app on http://localhost:5000 (waitress, production-grade WSGI server).
REM Only reachable from this computer. For internet use see docs/DEPLOYMENT.md (HTTPS required).
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.bat first.
    exit /b 1
)
echo Open http://localhost:5000 in your browser. Press Ctrl+C to stop.
".venv\Scripts\waitress-serve.exe" --listen=127.0.0.1:5000 --threads=4 wsgi:app
endlocal
