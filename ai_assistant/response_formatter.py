"""
Standardized JSON Response Formatter.
Ensures a stable contract for backend and frontend integration.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class AssistantResponse(BaseModel):
    answer: str = Field(..., description="Human-readable executive explanation")
    intent: str = Field(..., description="Detected intent category")
    status: str = Field("success", description="Status code: success or error")
    context_used: Dict[str, Any] = Field(default_factory=dict, description="Resolved context parameters")
    sources: List[str] = Field(default_factory=list, description="Targeted Supabase source tables")
    data: Dict[str, Any] = Field(default_factory=dict, description="Exact verified numbers extracted")
    warnings: List[str] = Field(default_factory=list, description="Data availability or boundary warnings")

def format_response(
    answer: str,
    intent: str,
    status: str = "success",
    context_used: Optional[Dict[str, Any]] = None,
    sources: Optional[List[str]] = None,
    data: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Generates the standardized response dictionary."""
    resp = AssistantResponse(
        answer=answer,
        intent=intent,
        status=status,
        context_used=context_used or {},
        sources=sources or [],
        data=data or {},
        warnings=warnings or []
    )
    return resp.model_dump()
