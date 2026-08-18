"""
Backend Teammate Integration Guide & Examples.
Shows how the main backend application can consume the AI Assistant module via Python import or HTTP API.
"""

import sys
from pathlib import Path

# Ensure project root is in python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ==============================================================================
# OPTION 1: DIRECT PYTHON IMPORT (Fastest, zero network overhead)
# ==============================================================================
def example_direct_python_import():
    from ai_assistant import ask

    user_question = "Why is ACO A1001 classified as High Performing?"
    dashboard_context = {
        "aco_id": "A1001",
        "year": 2021,
        "page": "ACO Explorer",
        "section": "Performance Drivers",
        "selected_metric": "Savings"
    }

    # Execute assistant
    result = ask(
        question=user_question,
        context=dashboard_context
    )

    print("--- Option 1 Output ---")
    print(f"Intent: {result['intent']}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    # Backend returns `result` or `result['answer']` to React frontend

# ==============================================================================
# OPTION 2: STANDALONE FASTAPI HTTP REQUEST (Decoupled microservice)
# ==============================================================================
def example_http_api_call():
    import requests

    endpoint = "http://localhost:8005/api/assistant/ask"
    payload = {
        "question": "What were the total expenditures for A1001 in 2021?",
        "context": {
            "aco_id": "A1001",
            "year": 2021
        }
    }

    try:
        response = requests.post(endpoint, json=payload, timeout=30)
        data = response.json()
        print("\n--- Option 2 Output ---")
        print(f"HTTP Status: {response.status_code}")
        print(f"Answer: {data.get('answer')}")
    except requests.exceptions.ConnectionError:
        print("\n--- Option 2 Output ---")
        print("Note: FastAPI server on port 8005 is not currently running. Start it with: uvicorn ai_assistant.api:app --port 8005")

if __name__ == "__main__":
    example_direct_python_import()
    example_http_api_call()
