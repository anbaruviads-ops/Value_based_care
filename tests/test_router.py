"""
Unit tests for Intent Classification and Selective Table Routing.
"""

from ai_assistant.router import classify_intent, IntentType

def test_financial_routing():
    intent, tables = classify_intent("What was ACO A1001's savings and expenditure in 2021?")
    assert intent == IntentType.FINANCIAL
    assert "fact_aco_performance" in tables

def test_quality_routing():
    intent, tables = classify_intent("What was the quality score and CAHPS rating for A1001?")
    assert intent == IntentType.QUALITY
    assert "aco_analytics" in tables

def test_utilization_routing():
    intent, tables = classify_intent("What was the emergency department utilization rate?")
    assert intent == IntentType.UTILIZATION
    assert "aco_utilization_ml_features" in tables

def test_anomaly_routing():
    intent, tables = classify_intent("Are there any anomalous metrics or alerts for A1199?")
    assert intent == IntentType.ANOMALY_RISK
    assert "aco_anomalies" in tables or "aco_alert" in tables

def test_segmentation_routing():
    intent, tables = classify_intent("Which segment is ACO A1001 in?")
    assert intent == IntentType.SEGMENTATION
    assert "aco_segmentation_results" in tables

def test_whatif_routing():
    intent, tables = classify_intent("What happens if ER utilization drops by 5%?", context={"scenario": {"a": 1}})
    assert intent == IntentType.WHAT_IF
    assert "what_if_simulator_engine" in tables

def test_provider_routing():
    intent, tables = classify_intent("Show me provider performance for Dr. Thota in ACO_005")
    assert intent == IntentType.PROVIDER_ANALYTICS
    assert "bhavya_provider_aco_features_final" in tables

def test_dashboard_context_metric_routing():
    intent, tables = classify_intent("Why is this metric red?", context={"section": "Performance Drivers", "selected_metric": "Utilization"})
    assert intent in [IntentType.UTILIZATION, IntentType.PERFORMANCE_DRIVER]
