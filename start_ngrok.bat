@echo off
echo ===================================================
echo   Share Local Ollama Globally with Ngrok Tunnel
echo ===================================================
echo.
echo If you have not added your free Ngrok auth token yet,
echo run: ngrok config add-authtoken <YOUR_TOKEN>
echo (Get your free token at: https://dashboard.ngrok.com/get-started/your-authtoken)
echo.
echo Starting Ngrok Tunnel on Port 11434...
ngrok http 11434
pause
