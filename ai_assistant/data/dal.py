"""
Data Access Layer (DAL) for ACO Healthcare Analytics.
Encapsulates table querying logic and schema normalization.
"""

from typing import Any, Dict, List, Optional
from ai_assistant.data.supabase_client import execute_read_query

def get_fact_aco_performance(aco_id: str, year: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieves CMS Fact ACO performance data for an ACO."""
    filters: Dict[str, Any] = {"ACO_ID": aco_id}
    if year is not None:
        filters["performance_year"] = year
    return execute_read_query("fact_aco_performance", filters=filters, order_by="performance_year", order_desc=True)

def get_aco_analytics(aco_id: str, year: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieves quality, demographic, and risk analytics from aco_analytics."""
    filters: Dict[str, Any] = {"aco_id": aco_id}
    if year is not None:
        filters["performance_year"] = year
    return execute_read_query("aco_analytics", filters=filters, order_by="performance_year", order_desc=True)

def get_aco_alerts(aco_id: str, year: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieves anomaly alerts from aco_alert."""
    filters: Dict[str, Any] = {"ACO_ID": aco_id}
    if year is not None:
        filters["performance_year"] = year
    return execute_read_query("aco_alert", filters=filters)

def get_aco_anomalies(aco_id: str, year: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieves anomaly detection outputs from aco_anomalies."""
    filters: Dict[str, Any] = {"ACO_ID": aco_id}
    if year is not None:
        filters["performance_year"] = year
    return execute_read_query("aco_anomalies", filters=filters)

def get_aco_anomaly_features(aco_id: str, year: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieves anomaly features (YoY changes in utilization/expenditure/quality)."""
    filters: Dict[str, Any] = {"ACO_ID": aco_id}
    if year is not None:
        filters["performance_year"] = year
    return execute_read_query("aco_anomaly_features", filters=filters)

def get_aco_risk_scores(aco_id: str, year: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieves ML risk scores from aco_risk_scores."""
    filters: Dict[str, Any] = {"ACO_ID": aco_id}
    if year is not None:
        filters["performance_year"] = year
    return execute_read_query("aco_risk_scores", filters=filters)

def get_aco_segmentation(aco_id: str, year: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieves segmentation cluster and performance segment from aco_segmentation_results."""
    filters: Dict[str, Any] = {"ACO_ID": aco_id}
    if year is not None:
        filters["performance_year"] = year
    return execute_read_query("aco_segmentation_results", filters=filters, order_by="performance_year", order_desc=True)

def get_aco_segmentation_features(aco_id: str, year: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieves ML features used in segmentation from aco_segmentation_ml_features."""
    filters: Dict[str, Any] = {"ACO_ID": aco_id}
    if year is not None:
        filters["performance_year"] = year
    return execute_read_query("aco_segmentation_ml_features", filters=filters, order_by="performance_year", order_desc=True)

def get_aco_utilization_features(aco_id: str, year: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieves detailed utilization metrics and scores from aco_utilization_ml_features."""
    filters: Dict[str, Any] = {"ACO_ID": aco_id}
    if year is not None:
        filters["performance_year"] = year
    return execute_read_query("aco_utilization_ml_features", filters=filters, order_by="performance_year", order_desc=True)

def get_aco_financial_ml_training(aco_id: str, year: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieves financial features and benchmarks from aco_financial_ml_training."""
    filters: Dict[str, Any] = {"ACO_ID": aco_id}
    if year is not None:
        # Check target_year or feature_year
        filters["target_year"] = year
    return execute_read_query("aco_financial_ml_training", filters=filters)

def get_provider_features(aco_id: Optional[str] = None, npi: Optional[int] = None, year: Optional[int] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieves provider-level drill-down data from bhavya_provider_aco_features_final."""
    filters: Dict[str, Any] = {}
    if aco_id:
        filters["aco_id"] = aco_id
    if npi:
        filters["rndrng_npi"] = npi
    if year:
        filters["year"] = year
    return execute_read_query("bhavya_provider_aco_features_final", filters=filters, limit=limit)

def get_service_metrics(aco_id: str, year: Optional[int] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieves service & HCPCS level metrics from service_metrics."""
    filters: Dict[str, Any] = {"ACO_ID": aco_id}
    if year is not None:
        filters["Year"] = year
    return execute_read_query("service_metrics", filters=filters, limit=limit)

def get_all_segmented_acos(segment: Optional[str] = None, year: Optional[int] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieves peer ACOs in a segment for peer benchmarking."""
    filters: Dict[str, Any] = {}
    if segment:
        filters["performance_segment"] = segment
    if year:
        filters["performance_year"] = year
    return execute_read_query("aco_segmentation_results", filters=filters, limit=limit)
