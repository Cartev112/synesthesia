# Synesthesia Backend Startup Script
# Run this from the project root directory

# Ensure we're in the project root
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

# Activate virtual environment if it exists
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Green
    .\venv\Scripts\Activate.ps1
} else {
    Write-Host "Warning: Virtual environment not found at .\venv" -ForegroundColor Yellow
}

# Add project root to PYTHONPATH
$env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"

# Start the backend server
Write-Host "Starting Synesthesia backend server..." -ForegroundColor Green
Write-Host "API will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""

python -m backend.api.main
