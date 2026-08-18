"""
Performance Driver Data Extraction Tools.
Synthesizes verified data points from aco_analytics, aco_utilization_ml_features, and aco_segmentation_ml_features.
"""

from typing import Any, Dict, List, Optional
from ai_assistant.data import dal

def get_performance_drivers(aco_id: str, year: Optional[int] = None) -> Dict[str, Any]:
    """
    Retrieves key performance drivers across Financial, Quality, and Utilization dimensions.
    """
    sources: List[str] = []
    drivers: Dict[str, Any] = {
        "aco_id": aco_id,
        "year": year,
        "financial_drivers": {},
        "quality_drivers": {},
        "utilization_drivers": {},
        "identified_attention_areas": []
    }

    # 1. Quality drivers
    quality_records = dal.get_aco_analytics(aco_id, year)
    if quality_records:
        sources.append("aco_analytics")
        q = quality_records[0]
        drivers["quality_drivers"] = {
            "quality_score": q.get("quality_score"),
            "quality_change_yoy": q.get("quality_change_yoy"),
            "quality_gap_to_100": q.get("quality_gap_to_100"),
            "category": q.get("quality_performance_category")
        }

    # 2. Utilization drivers
    util_records = dal.get_aco_utilization_features(aco_id, year)
    if util_records:
        sources.append("aco_utilization_ml_features")
        u = util_records[0]
        drivers["utilization_drivers"] = {
            "utilization_category": u.get("utilization_category"),
            "utilization_score": u.get("utilization_score"),
            "ed_utilization_change_yoy": u.get("ed_utilization_change_yoy"),
            "admission_change_yoy": u.get("admission_change_yoy"),
            "em_utilization_change_yoy": u.get("em_utilization_change_yoy"),
            "advanced_imaging_change_yoy": u.get("advanced_imaging_change_yoy"),
            "high_utilization_flag": u.get("high_utilization_flag")
        }

    # 3. Financial drivers
    seg_records = dal.get_aco_segmentation_features(aco_id, year)
    if seg_records:
        sources.append("aco_segmentation_ml_features")
        s = seg_records[0]
        drivers["financial_drivers"] = {
            "savings_loss_pct": s.get("SavingsLossPct"),
            "expenditure_variance_pct": s.get("ExpenditureVariancePct"),
            "pmpm": s.get("PMPM"),
            "benchmark_pmpm": s.get("BenchmarkPMPM"),
            "financial_gap": s.get("FinancialGap"),
            "gen_save_loss_yoy_pct": s.get("GenSaveLossYoYPct")
        }

    return {"data": drivers, "sources": sources}
