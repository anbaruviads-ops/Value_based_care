# 🚀 Backend Developer Integration Guide for AI Assistant Module

This guide provides step-by-step instructions for the **Main Backend Developer** to integrate the Standalone ACO AI Assistant module and expose it to the React frontend.

---

## 📌 Overview

The AI Assistant module is self-contained. It retrieves verified analytical and ML data from Supabase tables (e.g. `fact_aco_performance`, `aco_analytics`, `aco_anomalies`, `aco_segmentation_results`) and uses Ollama to generate grounded, executive responses.

---

## 📁 What You Need from This Module

If integrating via code repository, you only need the `ai_assistant/` folder and `requirements.txt`:
```
your-backend-project/
├── ai_assistant/        # Copy or import this entire folder
├── requirements.txt     # Add dependencies to your main backend
└── .env                 # Add environment variables
```

---

## ⚙️ Step 1: Add Environment Variables

In your backend `.env` file, include:

```ini
# Supabase Database
SUPABASE_URL=https://wuazaratvveeemdpeait.supabase.co
SUPABASE_KEY=your-supabase-key

# Ollama LLM
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

*(Note: If Ollama runs on a shared server, set `OLLAMA_BASE_URL=http://<server-ip>:11434`)*

---

## 🔌 Step 2: Choose Your Integration Method

### Method 1: Direct Python Import (Recommended — Fastest, Zero Network Latency)

In your main backend (e.g. `main.py` or your route handler):

```python
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional, Dict, Any

# Import the AI assistant function directly
from ai_assistant import ask

app = FastAPI()

class ChatRequest(BaseModel):
    question: str
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Called by the React frontend chatbot UI.
    """
    result = ask(
        question=request.question,
        context=request.context,
        session_id=request.session_id
    )
    
    # Return structured result to React frontend
    return {
        "answer": result["answer"],
        "intent": result["intent"],
        "sources": result["sources"],
        "status": result["status"]
    }
```

---

### Method 2: HTTP Microservice Call (If running as a separate service)

If the AI Assistant is running on port `8005`:

```python
import httpx
from fastapi import FastAPI, HTTPException

app = FastAPI()

AI_ASSISTANT_URL = "http://localhost:8005/api/assistant/ask"

@app.post("/api/chat")
async def proxy_chat(request_data: dict):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(AI_ASSISTANT_URL, json=request_data, timeout=60.0)
            return response.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"AI Assistant service unreachable: {exc}")
```

---

## 💻 Step 3: Frontend Request & Response Contract

### What the React Frontend Sends:
```json
{
  "question": "Why is ACO A1001 classified as High Performing?",
  "context": {
    "aco_id": "A1001",
    "year": 2021,
    "page": "ACO Explorer",
    "section": "Performance Drivers",
    "selected_metric": "Savings"
  }
}
```

### What the React Frontend Receives:
```json
{
  "answer": "Based on verified data from aco_segmentation_results, ACO A1001 is classified as **High Performing** (Cluster 0) with a **7.17% savings rate** and a perfect **100.0% Quality Score**.",
  "intent": "SEGMENTATION",
  "status": "success",
  "sources": [
    "aco_segmentation_results",
    "aco_segmentation_ml_features"
  ]
}
```
*The React UI should display `answer` in the chatbot bubble.*
