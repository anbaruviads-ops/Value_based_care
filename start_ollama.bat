@echo off
echo ===================================================
echo   Starting Ollama Service and Pulling Llama3.2
echo ===================================================

where ollama >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Ollama is not installed or not in PATH.
    echo Please install Ollama from https://ollama.com/download
    pause
    exit /b 1
)

echo [1/2] Ensuring Ollama model llama3.2 is available...
ollama pull llama3.2

echo.
echo [2/2] Starting Ollama serve...
start "" ollama serve

echo [SUCCESS] Ollama service is deployed and running on http://localhost:11434
