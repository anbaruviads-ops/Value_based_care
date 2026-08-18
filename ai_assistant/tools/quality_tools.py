"""
Quality Data Extraction Tools.
Sources: aco_analytics, fact_aco_performance
"""

from typing import Any, Dict, List, Optional
from ai_assistant.data import dal

def get_quality_metrics(aco_id: str, year: Optional[int] = None) -> Dict[str, Any]:
    """
    Extracts quality scores, CAHPS patient experience measures, and population demographics.
    """
    sources: List[str] = []
    data: Dict[str, Any] = {}

    analytics_records = dal.get_aco_analytics(aco_id, year)
    if analytics_records:
        sources.append("aco_analytics")
        rec = analytics_records[0]
        data = {
            "aco_id": rec.get("aco_id"),
            "aco_name": rec.get("aco_name"),
            "year": rec.get("performance_year"),
            "quality_score": rec.get("quality_score"),
            "previous_quality_score": rec.get("previous_quality_score"),
            "quality_change_yoy": rec.get("quality_change_yoy"),
            "quality_change_yoy_pct": rec.get("quality_change_yoy_pct"),
            "quality_gap_to_100": rec.get("quality_gap_to_100"),
            "quality_category": rec.get("quality_performance_category"),
            "cahps_scores": {
                "getting_timely_care": rec.get("cahps_1"),
                "provider_communication": rec.get("cahps_2"),
                "patient_rating_of_provider": rec.get("cahps_3"),
                "access_to_specialists": rec.get("cahps_4"),
                "health_promotion_education": rec.get("cahps_5"),
                "shared_decision_making": rec.get("cahps_6"),
                "health_status_functional": rec.get("cahps_7"),
                "stewardship_of_resources": rec.get("cahps_8"),
                "care_coordination": rec.get("cahps_9"),
                "courteous_staff": rec.get("cahps_11")
            },
            "demographics": {
                "assigned_beneficiaries": rec.get("assigned_beneficiaries"),
                "female_pct": rec.get("female_pct"),
                "male_pct": rec.get("male_pct"),
                "disabled_pct": rec.get("disabled_pct"),
                "esrd_pct": rec.get("esrd_pct"),
                "dual_eligible_pct": rec.get("dual_eligible_pct"),
                "risk_profile_category": rec.get("risk_profile_category"),
                "average_risk_score": rec.get("average_available_risk_score")
            }
        }
        return {"data": data, "sources": sources, "records": analytics_records}

    # Fallback to fact_aco_performance
    fact_records = dal.get_fact_aco_performance(aco_id, year)
    if fact_records:
        sources.append("fact_aco_performance")
        rec = fact_records[0]
        data = {
            "aco_id": rec.get("ACO_ID"),
            "aco_name": rec.get("ACO_Name"),
            "year": rec.get("performance_year"),
            "quality_score": rec.get("QualScore"),
            "assigned_beneficiaries": rec.get("N_AB")
        }
        return {"data": data, "sources": sources, "records": fact_records}

    return {"data": {}, "sources": sources, "records": []}

def get_quality_history(aco_id: str) -> Dict[str, Any]:
    """Retrieves longitudinal quality history across all recorded years."""
    records = dal.get_aco_analytics(aco_id)
    sources = ["aco_analytics"]
    
    if not records:
        records = dal.get_fact_aco_performance(aco_id)
        sources = ["fact_aco_performance"]

    history = []
    for r in records:
        history.append({
            "year": r.get("performance_year"),
            "quality_score": r.get("quality_score") or r.get("QualScore"),
            "quality_category": r.get("quality_performance_category"),
            "beneficiaries": r.get("assigned_beneficiaries") or r.get("N_AB")
        })

    history = sorted(history, key=lambda x: x["year"] if x["year"] else 0)
    return {"history": history, "sources": sources}
