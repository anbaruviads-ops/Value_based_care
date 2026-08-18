"""
Context Manager & Conversation State Resolver.
Extracts entities (ACO_ID, Year, Comparisons) from question and resolves dashboard context with short-term memory.
"""

import re
from typing import Any, Dict, List, Optional

class ConversationMemory:
    """Simple in-memory session tracker for multi-turn conversational follow-ups."""
    def __init__(self):
        self._history: Dict[str, Dict[str, Any]] = {}

    def get_last_context(self, session_id: str) -> Dict[str, Any]:
        return self._history.get(session_id, {})

    def update(self, session_id: str, context: Dict[str, Any]):
        if session_id:
            current = self._history.get(session_id, {})
            current.update(context)
            self._history[session_id] = current

memory_store = ConversationMemory()

def resolve_context(
    question: str,
    dashboard_context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Combines user question entity extraction, ambient dashboard context, and prior session memory.
    Rule: Explicit entities in the question ALWAYS override ambient dashboard context.
    """
    resolved: Dict[str, Any] = {
        "aco_id": None,
        "year": None,
        "compare_with_aco_id": None,
        "page": None,
        "section": None,
        "selected_metric": None,
        "scenario": None
    }

    # 1. Incorporate prior session memory
    if session_id:
        prior = memory_store.get_last_context(session_id)
        resolved.update({k: v for k, v in prior.items() if v is not None})

    # 2. Incorporate ambient dashboard context
    if dashboard_context:
        for k in ["aco_id", "year", "page", "section", "selected_metric", "scenario", "result"]:
            if dashboard_context.get(k) is not None:
                resolved[k] = dashboard_context[k]

    # 3. Extract explicit ACO IDs from Question (e.g. A1001, ACO-101, ACO_005)
    aco_matches = re.findall(r'\b(A\d{3,5}|ACO[_\-\s]?\d{1,5})\b', question, re.IGNORECASE)
    if aco_matches:
        # Standardize format (e.g., ACO-101 -> A101 or keep clean)
        cleaned_matches = [m.replace("-", "").replace("_", "").replace(" ", "").upper() for m in aco_matches]
        resolved["aco_id"] = cleaned_matches[0]
        if len(cleaned_matches) > 1:
            resolved["compare_with_aco_id"] = cleaned_matches[1]

    # 4. Extract explicit Year from Question (e.g., 2020, 2021, 2022, 2023, 2024)
    year_match = re.search(r'\b(201\d|202\d)\b', question)
    if year_match:
        resolved["year"] = int(year_match.group(1))

    # 5. Update session memory with newly resolved state
    if session_id:
        memory_store.update(session_id, {
            "aco_id": resolved["aco_id"],
            "year": resolved["year"]
        })

    return resolved
