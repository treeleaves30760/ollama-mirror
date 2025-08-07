# Test Ollama Mirror Server (PowerShell)
Write-Host "=== Testing Ollama Mirror Server ===" -ForegroundColor Green
Write-Host ""

# Set environment variable for this session
$env:OLLAMA_HOST = "http://localhost:11434"
Write-Host "✓ Set OLLAMA_HOST to: $env:OLLAMA_HOST" -ForegroundColor Green

# Test if server is running
Write-Host "Testing server health..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "✓ Server is running!" -ForegroundColor Green
    } else {
        Write-Host "✗ Server responded with status: $($response.StatusCode)" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "✗ ERROR: Server is not running on http://localhost:11434" -ForegroundColor Red
    Write-Host "Please start the server first using .\start_server.ps1" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Run Python tests
Write-Host "Running test suite..." -ForegroundColor Cyan
python test_mirror.py

Write-Host ""
Write-Host "Test completed. Check the output above for results." -ForegroundColor Yellow
Read-Host "Press Enter to exit"
