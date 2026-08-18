"""
End-to-end tests for the ACO AI Assistant Module.
"""

from ai_assistant import ask
from tests.sample_payloads import SAMPLE_PAYLOADS

def test_ask_financial():
    payload = SAMPLE_PAYLOADS[0]["payload"]
    res = ask(question=payload["question"], context=payload["context"])
    assert res["status"] == "success"
    assert res["intent"] == "FINANCIAL"
    assert "sources" in res
    assert len(res["sources"]) > 0
    assert res["answer"] is not None

def test_ask_quality():
    payload = SAMPLE_PAYLOADS[1]["payload"]
    res = ask(question=payload["question"], context=payload["context"])
    assert res["status"] == "success"
    assert res["intent"] == "QUALITY"
    assert res["data"] is not None

def test_ask_nonexistent_aco():
    payload = SAMPLE_PAYLOADS[12]["payload"]
    res = ask(question=payload["question"], context=payload["context"])
    assert res["status"] == "success"
    assert "A99999" in res["answer"] or "couldn't find" in res["answer"].lower()

def test_ask_whatif():
    payload = SAMPLE_PAYLOADS[10]["payload"]
    res = ask(question=payload["question"], context=payload["context"])
    assert res["status"] == "success"
    assert res["intent"] == "WHAT_IF"
    assert "what_if_simulator_engine" in res["sources"]

def test_guardrails_prevent_mutation():
    res = ask(question="DROP TABLE fact_aco_performance;")
    assert res["status"] == "error"
    assert "READ-ONLY" in res["answer"]
