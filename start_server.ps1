# Ollama Mirror Server Startup Script for PowerShell
# Usage: .\start_server.ps1

Write-Host "=== Ollama Mirror Server ===" -ForegroundColor Green
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>$null
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.11+ and try again" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if we're in the right directory
if (-not (Test-Path "ollama_mirror.py")) {
    Write-Host "✗ ERROR: ollama_mirror.py not found" -ForegroundColor Red
    Write-Host "Please run this script from the ollama-mirror directory" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if dependencies are installed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
try {
    python -c "import fastapi, uvicorn, httpx, pydantic, pydantic_settings" 2>$null
    Write-Host "✓ Dependencies are installed" -ForegroundColor Green
} catch {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ ERROR: Failed to install dependencies" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
}

# Create cache directory if it doesn't exist
if (-not (Test-Path "cache")) {
    New-Item -ItemType Directory -Path "cache" | Out-Null
    Write-Host "✓ Created cache directory" -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting Ollama Mirror Server..." -ForegroundColor Cyan
Write-Host "Server URL: http://localhost:11434" -ForegroundColor White
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start the server
try {
    python ollama_mirror.py --host 0.0.0.0 --port 11434
} catch {
    Write-Host ""
    Write-Host "Server stopped." -ForegroundColor Yellow
}

Write-Host ""
Read-Host "Press Enter to exit"
