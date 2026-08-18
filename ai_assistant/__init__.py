"""
Standalone AI Assistant Module for ACO Healthcare Analytics & Value-Based Care.
"""

from ai_assistant.assistant import ask, ACOAssistant
from ai_assistant.response_formatter import AssistantResponse
from ai_assistant.api import app

__all__ = ["ask", "ACOAssistant", "AssistantResponse", "app"]
__version__ = "1.0.0"
