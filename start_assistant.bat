@echo off
echo ===================================================
echo   Starting Standalone AI Assistant FastAPI Server
echo ===================================================

python -m uvicorn ai_assistant.api:app --host 0.0.0.0 --port 8005 --reload
