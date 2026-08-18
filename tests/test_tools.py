"""
Unit tests for data extraction tools and Data Access Layer.
"""

import pytest
from ai_assistant import tools

def test_financial_tools():
    res = tools.get_financial_metrics("A1001", 2021)
    assert res is not None
    assert "fact_aco_performance" in res["sources"] or "aco_financial_ml_training" in res["sources"]
    data = res["data"]
    assert data["aco_id"] == "A1001"
    assert data["total_expenditure"] is not None

def test_quality_tools():
    res = tools.get_quality_metrics("A1001", 2021)
    assert res is not None
    assert "aco_analytics" in res["sources"] or "fact_aco_performance" in res["sources"]
    data = res["data"]
    assert data["quality_score"] == 100.0
    assert "cahps_scores" in data or "quality_score" in data

def test_utilization_tools():
    res = tools.get_utilization_metrics("A1001", 2021)
    assert res is not None
    data = res["data"]
    assert data["ed_visits"] is not None

def test_segmentation_tools():
    res = tools.get_segmentation_results("A1001", 2021)
    assert res is not None
    data = res["data"]
    assert data["performance_segment"] == "High Performing"

def test_anomaly_tools():
    res = tools.get_anomaly_and_risk("A1199", 2023)
    assert res is not None
    data = res["data"]
    assert len(data["alerts"]) > 0 or len(data["anomalies"]) > 0

def test_provider_tools():
    res = tools.get_provider_details(aco_id="ACO_005")
    assert res is not None
    data = res["data"]
    assert len(data["providers"]) > 0

def test_service_tools():
    res = tools.get_service_analytics("A1001", 2021)
    assert res is not None
    data = res["data"]
    assert len(data["services"]) > 0

def test_whatif_tools():
    context = {
        "scenario": {"metric": "ER utilization", "change": "-5%"},
        "result": {"projected_expenditure_change": "-2.1%"}
    }
    res = tools.interpret_whatif_scenario(context)
    assert res["data"]["type"] == "SIMULATED_SCENARIO"
    assert "what_if_simulator_engine" in res["sources"]
