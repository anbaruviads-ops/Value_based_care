"""
Service & HCPCS Code Data Extraction Tools.
Sources: service_metrics
"""

from typing import Any, Dict, List, Optional
from ai_assistant.data import dal

def get_service_analytics(aco_id: str, year: Optional[int] = None) -> Dict[str, Any]:
    """
    Extracts high-cost / high-utilization service categories and HCPCS codes.
    """
    sources = ["service_metrics"]
    records = dal.get_service_metrics(aco_id=aco_id, year=year, limit=10)

    services = []
    for r in records:
        services.append({
            "aco_id": r.get("ACO_ID"),
            "year": r.get("Year"),
            "hcpcs_code": r.get("HCPCS_Cd"),
            "description": r.get("HCPCS_Desc"),
            "category": r.get("service_category"),
            "provider_count": r.get("provider_count"),
            "beneficiary_count": r.get("beneficiary_count"),
            "service_volume": r.get("service_volume"),
            "total_payment": r.get("total_payment"),
            "avg_payment_per_beneficiary": r.get("avg_payment_per_beneficiary"),
            "high_cost_flag": r.get("high_cost_service"),
            "high_utilization_flag": r.get("high_utilization_service"),
            "segment": r.get("service_performance_segment")
        })

    return {"data": {"services": services, "count": len(services)}, "sources": sources}
