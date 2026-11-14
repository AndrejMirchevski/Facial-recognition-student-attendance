@echo off
REM One-time setup: virtual environment, packages, model weights, database.
REM Requires Python 3.11 (64-bit) installed from python.org.
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher "py" not found. Install Python 3.11 from https://www.python.org/downloads/
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment .venv ...
    py -3.11 -m venv .venv || (echo Could not create venv - is Python 3.11 installed? & exit /b 1)
)

echo Installing packages ...
".venv\Scripts\python.exe" -m pip install --upgrade pip || exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements.txt || exit /b 1

echo Downloading models ...
".venv\Scripts\python.exe" scripts\download_models.py || exit /b 1

echo Checking models ...
".venv\Scripts\python.exe" scripts\check_models.py || exit /b 1

echo Creating the database ...
".venv\Scripts\python.exe" scripts\init_db.py || exit /b 1

echo.
echo Setup complete.
echo Next: create the first professor account with
echo     .venv\Scripts\python scripts\create_professor.py
echo then run start_server.bat
endlocal
