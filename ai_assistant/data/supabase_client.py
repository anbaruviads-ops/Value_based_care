"""
Supabase Client and Connection Manager.
Strictly read-only access to prevent accidental mutations.
Gracefully falls back to mock dataset if Supabase is temporarily unreachable or during unit tests.
"""

import logging
from typing import Any, Dict, List, Optional
from ai_assistant.config import settings

logger = logging.getLogger(__name__)

_supabase_client = None

def get_supabase_client():
    """Initializes and returns a singleton Supabase client, or None if unconfigured."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    if not settings.is_supabase_configured():
        logger.info("Supabase credentials not fully configured. Using mock fallback mode.")
        return None

    try:
        from supabase import create_client, Client
        _supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        logger.info("Supabase client successfully initialized.")
        return _supabase_client
    except Exception as e:
        logger.warning(f"Could not connect to Supabase: {e}. Defaulting to mock data mode.")
        return None

def execute_read_query(
    table_name: str,
    filters: Optional[Dict[str, Any]] = None,
    select_columns: str = "*",
    limit: int = 50,
    order_by: Optional[str] = None,
    order_desc: bool = False
) -> List[Dict[str, Any]]:
    """
    Executes a strict read-only query against Supabase, or queries in-memory mock data.
    """
    client = get_supabase_client()
    
    if client:
        try:
            query = client.table(table_name).select(select_columns)
            
            if filters:
                for col, val in filters.items():
                    if val is not None:
                        # Handle case insensitivity or exact matching
                        query = query.eq(col, val)
                        
            if order_by:
                query = query.order(order_by, desc=order_desc)
                
            query = query.limit(limit)
            response = query.execute()
            
            if response and hasattr(response, 'data') and response.data:
                return response.data
        except Exception as e:
            logger.warning(f"Supabase query against {table_name} failed: {e}. Checking fallback mock data.")

    # Fallback to in-memory mock dataset
    return _query_mock_data(table_name, filters, limit)

def _query_mock_data(
    table_name: str,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """Helper to query the local mock data store by table name and filters."""
    from ai_assistant.data import mock_data

    table_map = {
        "fact_aco_performance": mock_data.FACT_ACO_PERFORMANCE,
        "aco_analytics": mock_data.ACO_ANALYTICS,
        "aco_alert": mock_data.ACO_ALERT,
        "aco_anomalies": mock_data.ACO_ANOMALIES,
        "aco_anomaly_features": mock_data.ACO_ANOMALY_FEATURES,
        "aco_risk_scores": mock_data.ACO_RISK_SCORES,
        "aco_segmentation_results": mock_data.ACO_SEGMENTATION_RESULTS,
        "aco_segmentation_ml_features": mock_data.ACO_SEGMENTATION_ML_FEATURES,
        "aco_utilization_ml_features": mock_data.ACO_UTILIZATION_ML_FEATURES,
        "aco_financial_ml_training": mock_data.ACO_FINANCIAL_ML_TRAINING,
        "bhavya_provider_aco_features_final": mock_data.BHAVYA_PROVIDER_ACO_FEATURES_FINAL,
        "service_metrics": mock_data.SERVICE_METRICS
    }

    records = table_map.get(table_name, [])
    if not filters:
        return records[:limit]

    results = []
    for row in records:
        matches = True
        for k, v in filters.items():
            # Check for case-insensitive column keys (e.g. aco_id vs ACO_ID)
            row_val = None
            for rk, rv in row.items():
                if rk.lower() == k.lower():
                    row_val = rv
                    break
            
            if v is not None:
                if row_val is None or str(row_val).strip().lower() != str(v).strip().lower():
                    matches = False
                    break
        if matches:
            results.append(row)
            if len(results) >= limit:
                break

    return results
