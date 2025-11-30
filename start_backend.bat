@echo off
REM Synesthesia Backend Startup Script (Windows CMD)
REM Run this from the project root directory

cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found at .\venv
)

REM Add project root to PYTHONPATH
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Start the backend server
echo Starting Synesthesia backend server...
echo API will be available at: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.

python -m backend.api.main
