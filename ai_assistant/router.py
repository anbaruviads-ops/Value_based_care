"""
Intent Classification & Selective Table Router.
Routes questions to only relevant tables and tools based on question semantics and dashboard context.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

class IntentType:
    FINANCIAL = "FINANCIAL"
    QUALITY = "QUALITY"
    UTILIZATION = "UTILIZATION"
    HISTORICAL_TREND = "HISTORICAL_TREND"
    PERFORMANCE_DRIVER = "PERFORMANCE_DRIVER"
    SEGMENTATION = "SEGMENTATION"
    ANOMALY_RISK = "ANOMALY_RISK"
    PEER_COMPARISON = "PEER_COMPARISON"
    RECOMMENDATION = "RECOMMENDATION"
    PROVIDER_ANALYTICS = "PROVIDER_ANALYTICS"
    SERVICE_ANALYTICS = "SERVICE_ANALYTICS"
    WHAT_IF = "WHAT_IF"
    DASHBOARD_OVERVIEW = "DASHBOARD_OVERVIEW"
    GENERAL_DATA = "GENERAL_DATA"

TABLE_ROUTING_MAP = {
    IntentType.FINANCIAL: ["fact_aco_performance", "aco_financial_ml_training"],
    IntentType.QUALITY: ["aco_analytics", "fact_aco_performance"],
    IntentType.UTILIZATION: ["aco_utilization_ml_features", "fact_aco_performance"],
    IntentType.HISTORICAL_TREND: ["fact_aco_performance", "aco_analytics", "aco_financial_ml_training"],
    IntentType.PERFORMANCE_DRIVER: ["aco_analytics", "aco_utilization_ml_features", "aco_segmentation_ml_features"],
    IntentType.SEGMENTATION: ["aco_segmentation_results", "aco_segmentation_ml_features"],
    IntentType.ANOMALY_RISK: ["aco_anomalies", "aco_alert", "aco_risk_scores", "aco_anomaly_features"],
    IntentType.PEER_COMPARISON: ["aco_segmentation_results", "fact_aco_performance", "aco_financial_ml_training"],
    IntentType.RECOMMENDATION: ["aco_alert", "aco_risk_scores", "aco_segmentation_results", "aco_anomalies"],
    IntentType.PROVIDER_ANALYTICS: ["bhavya_provider_aco_features_final"],
    IntentType.SERVICE_ANALYTICS: ["service_metrics"],
    IntentType.WHAT_IF: ["what_if_simulator_engine"],
    IntentType.DASHBOARD_OVERVIEW: ["fact_aco_performance", "aco_analytics"],
    IntentType.GENERAL_DATA: ["fact_aco_performance", "aco_analytics"]
}

def classify_intent(question: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, List[str]]:
    """
    Classifies the user intent and returns the intent name and targeted Supabase tables.
    Prioritizes explicit user question content over ambient dashboard context.
    """
    q = question.lower()
    ctx = context or {}
    section = (ctx.get("section") or "").lower()
    page = (ctx.get("page") or "").lower()
    selected_metric = (ctx.get("selected_metric") or "").lower()

    # 1. What-if checks
    if ctx.get("scenario") or "what if" in q or "what happens if" in q or "simulate" in q or "hypothetical" in q:
        return IntentType.WHAT_IF, TABLE_ROUTING_MAP[IntentType.WHAT_IF]

    # 2. Recommendations
    if any(k in q for k in ["recommend", "focus on", "action plan", "suggest", "improve performance", "what should we do", "what should the payer"]):
        return IntentType.RECOMMENDATION, TABLE_ROUTING_MAP[IntentType.RECOMMENDATION]

    # 3. Anomaly & Risk / Alerts
    if any(k in q for k in ["anomal", "alert", "risk", "outlier", "flagged", "severity", "unusual", "spike"]):
        return IntentType.ANOMALY_RISK, TABLE_ROUTING_MAP[IntentType.ANOMALY_RISK]

    # 4. Peer Comparison
    if any(k in q for k in ["peer", "compare", "benchmark against", "other acos", "target"]):
        return IntentType.PEER_COMPARISON, TABLE_ROUTING_MAP[IntentType.PEER_COMPARISON]

    # 5. Segmentation & Clusters
    if any(k in q for k in ["segment", "cluster", "high attention", "moderate attention", "high performing", "low performing", "classification"]):
        return IntentType.SEGMENTATION, TABLE_ROUTING_MAP[IntentType.SEGMENTATION]

    # 6. Performance Drivers / Underperformance reasons
    if any(k in q for k in ["driver", "why is", "reason", "underperform", "why red", "cause", "explain drop", "contributing factor"]):
        return IntentType.PERFORMANCE_DRIVER, TABLE_ROUTING_MAP[IntentType.PERFORMANCE_DRIVER]

    # 7. Historical Trends & Multi-year
    if any(k in q for k in ["trend", "history", "over time", "from 20", "to 20", "longitudinal", "year over year", "yoy"]):
        return IntentType.HISTORICAL_TREND, TABLE_ROUTING_MAP[IntentType.HISTORICAL_TREND]

    # 8. Provider Level Analytics
    if any(k in q for k in ["provider", "doctor", "npi", "dr.", "specialist", "pcp", "physician"]):
        return IntentType.PROVIDER_ANALYTICS, TABLE_ROUTING_MAP[IntentType.PROVIDER_ANALYTICS]

    # 9. Service / HCPCS Analytics
    if any(k in q for k in ["service", "hcpcs", "cpt", "code", "imaging", "procedure"]):
        return IntentType.SERVICE_ANALYTICS, TABLE_ROUTING_MAP[IntentType.SERVICE_ANALYTICS]

    # 10. Quality & CAHPS
    if any(k in q for k in ["quality", "cahps", "patient satisfaction", "measure", "score", "preventive"]):
        return IntentType.QUALITY, TABLE_ROUTING_MAP[IntentType.QUALITY]

    # 11. Utilization (ED, Admissions, Inpatient, SNF)
    if any(k in q for k in ["utiliz", "ed visit", "emergency", "admission", "readmission", "snf", "inpatient", "er "]):
        return IntentType.UTILIZATION, TABLE_ROUTING_MAP[IntentType.UTILIZATION]

    # 12. Financial (Expenditure, Savings, Benchmark, PMPM)
    if any(k in q for k in ["expenditure", "spending", "benchmark", "saving", "loss", "pmpm", "revenue", "cost", "dollar", "$"]):
        return IntentType.FINANCIAL, TABLE_ROUTING_MAP[IntentType.FINANCIAL]

    # 13. Dashboard Context Fallbacks (When question is concise like "Why is this metric red?")
    if "quality" in selected_metric or "quality" in section:
        return IntentType.QUALITY, TABLE_ROUTING_MAP[IntentType.QUALITY]
    if "utilization" in selected_metric or "utilization" in section:
        return IntentType.UTILIZATION, TABLE_ROUTING_MAP[IntentType.UTILIZATION]
    if "financial" in selected_metric or "financial" in section or "savings" in selected_metric:
        return IntentType.FINANCIAL, TABLE_ROUTING_MAP[IntentType.FINANCIAL]
    if "driver" in section:
        return IntentType.PERFORMANCE_DRIVER, TABLE_ROUTING_MAP[IntentType.PERFORMANCE_DRIVER]
    if "anomaly" in section or "alert" in section:
        return IntentType.ANOMALY_RISK, TABLE_ROUTING_MAP[IntentType.ANOMALY_RISK]
    if "segment" in section:
        return IntentType.SEGMENTATION, TABLE_ROUTING_MAP[IntentType.SEGMENTATION]

    # 14. Dashboard Explanation
    if any(k in q for k in ["explain this dashboard", "what does this page show", "overview", "summary"]):
        return IntentType.DASHBOARD_OVERVIEW, TABLE_ROUTING_MAP[IntentType.DASHBOARD_OVERVIEW]

    # General Fallback
    return IntentType.GENERAL_DATA, TABLE_ROUTING_MAP[IntentType.GENERAL_DATA]
