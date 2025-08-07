@echo off
echo Testing Ollama Mirror Server...
echo.

REM Set Ollama to use the mirror
set OLLAMA_HOST=http://localhost:11434

echo Setting OLLAMA_HOST to: %OLLAMA_HOST%
echo.

REM Test if server is running
echo Testing server health...
curl -s http://localhost:11434/ > nul 2>&1
if errorlevel 1 (
    echo ERROR: Server is not running on http://localhost:11434
    echo Please start the server first using start_server.bat
    pause
    exit /b 1
)

echo ✓ Server is running!
echo.

REM Run tests
echo Running test suite...
python test_mirror.py

echo.
echo Test completed. Check the output above for results.
pause
