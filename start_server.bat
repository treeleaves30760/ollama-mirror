@echo off
echo ===================================
echo    Ollama Mirror Server (Windows)
echo ===================================
echo.

REM Stop any existing Ollama processes
echo Stopping existing Ollama processes...
taskkill /f /im ollama.exe 2>nul
taskkill /f /im "ollama app.exe" 2>nul
echo.

REM Check if Python is installed
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ and try again
    pause
    exit /b 1
)
echo ✓ Python is installed

REM Check if dependencies are installed
echo Checking dependencies...
python -c "import fastapi, uvicorn, httpx, pydantic, pydantic_settings" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt --upgrade
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)
echo ✓ Dependencies are ready

REM Create cache directory
if not exist cache mkdir cache
echo ✓ Cache directory ready

REM Start the server
echo.
echo ==========================================
echo   Starting Ollama Mirror Server...
echo   URL: http://localhost:11434
echo   Press Ctrl+C to stop the server
echo ==========================================
echo.
python cli.py start --host 0.0.0.0 --port 11434

echo.
echo Server stopped.
pause
