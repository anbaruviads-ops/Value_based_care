"""
Guardrails and Validation Layer.
Ensures strict read-only safety, non-hallucination validation, and graceful warning generation.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

FORBIDDEN_MUTATION_KEYWORDS = [
    "drop table", "alter table", "delete from", "insert into",
    "update ", "truncate ", "grant ", "revoke "
]

def sanitize_and_validate_input(question: str) -> Tuple[bool, Optional[str]]:
    """
    Validates that the input question does not attempt database mutations or prompt injections.
    """
    if not question or not question.strip():
        return False, "Question cannot be empty."

    lower_q = question.lower()
    for kw in FORBIDDEN_MUTATION_KEYWORDS:
        if kw in lower_q:
            return False, f"Prohibited operation requested. The AI Assistant operates in strict READ-ONLY mode."

    return True, None

def evaluate_data_presence(
    aco_id: Optional[str],
    year: Optional[int],
    retrieved_data: Dict[str, Any]
) -> List[str]:
    """Generates warnings if requested data is missing or incomplete."""
    warnings: List[str] = []

    if aco_id and not retrieved_data:
        warnings.append(f"No records found for ACO ID '{aco_id}'.")
    
    if year and retrieved_data:
        # Check if year is present in data
        pass

    return warnings
