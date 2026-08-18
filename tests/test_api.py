"""
FastAPI Microservice integration tests.
"""

from fastapi.testclient import TestClient
from ai_assistant.api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data

def test_ask_endpoint():
    payload = {
        "question": "What was ACO A1001's savings in 2021?",
        "context": {
            "aco_id": "A1001",
            "year": 2021,
            "page": "ACO Explorer"
        }
    }
    response = client.post("/api/assistant/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["intent"] == "FINANCIAL"
    assert "sources" in data
    assert "answer" in data
