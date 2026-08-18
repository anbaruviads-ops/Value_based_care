"""
FastAPI Microservice for the AI Assistant Module.
Allows the main backend or React frontend to communicate with the AI Assistant over HTTP REST.
"""

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ai_assistant.config import settings
from ai_assistant.assistant import ask, ACOAssistant
from ai_assistant.response_formatter import AssistantResponse
from ai_assistant.data.supabase_client import get_supabase_client

app = FastAPI(
    title="ACO Healthcare Analytics AI Assistant API",
    description="Standalone AI Assistant Microservice for Value-Based Care Command Centers",
    version="1.0.0"
)

# Enable CORS for local development and backend proxy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str = Field(..., json_schema_extra={"example": "What was ACO A1001's savings rate and expenditure in 2021?"})
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        json_schema_extra={
            "example": {
                "aco_id": "A1001",
                "year": 2021,
                "page": "ACO Explorer",
                "section": "Financial Summary",
                "selected_metric": "Savings"
            }
        }
    )
    session_id: Optional[str] = Field(default=None, json_schema_extra={"example": "user_session_123"})

@app.get("/health", tags=["System"])
def health_check():
    """Returns the operational health and connection status of the assistant service."""
    db_connected = get_supabase_client() is not None
    return {
        "status": "healthy",
        "service": "aco_ai_assistant",
        "supabase_connected": db_connected,
        "ollama_base_url": settings.OLLAMA_BASE_URL,
        "ollama_model": settings.OLLAMA_MODEL
    }

@app.post("/api/assistant/ask", response_model=AssistantResponse, tags=["Assistant"])
def ask_assistant(request: AskRequest):
    """
    Main Assistant Endpoint.
    Accepts user question + dashboard context and returns an executive structured response.
    """
    try:
        response_dict = ask(
            question=request.question,
            context=request.context,
            session_id=request.session_id
        )
        return response_dict
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal assistant processing error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "ai_assistant.api:app",
        host=settings.ASSISTANT_HOST,
        port=settings.ASSISTANT_PORT,
        reload=settings.DEBUG
    )
