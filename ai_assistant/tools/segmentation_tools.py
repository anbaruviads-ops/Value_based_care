"""
Segmentation Data Extraction Tools.
Sources: aco_segmentation_results, aco_segmentation_ml_features
"""

from typing import Any, Dict, List, Optional
from ai_assistant.data import dal

def get_segmentation_results(aco_id: str, year: Optional[int] = None) -> Dict[str, Any]:
    """
    Extracts ML segmentation results (Performance Segment, Cluster ID, Performance Index).
    """
    sources: List[str] = []
    data: Dict[str, Any] = {}

    seg_records = dal.get_aco_segmentation(aco_id, year)
    if seg_records:
        sources.append("aco_segmentation_results")
        rec = seg_records[0]
        data = {
            "aco_id": rec.get("ACO_ID"),
            "year": rec.get("performance_year"),
            "cluster_id": rec.get("cluster_id"),
            "performance_segment": rec.get("performance_segment"),
            "performance_index": rec.get("performance_index"),
            "savings_loss_pct": rec.get("SavingsLossPct"),
            "expenditure_variance_pct": rec.get("ExpenditureVariancePct"),
            "utilization_score": rec.get("utilization_score"),
            "quality_score": rec.get("quality_score")
        }

    # Enrich with segmentation feature details if available
    feat_records = dal.get_aco_segmentation_features(aco_id, year)
    if feat_records:
        sources.append("aco_segmentation_ml_features")
        frec = feat_records[0]
        data["feature_breakdown"] = {
            "pmpm": frec.get("PMPM"),
            "benchmark_pmpm": frec.get("BenchmarkPMPM"),
            "financial_gap": frec.get("FinancialGap"),
            "ed_visits_per_beneficiary": frec.get("ed_visits_per_beneficiary"),
            "admissions_per_beneficiary": frec.get("admissions_per_beneficiary"),
            "advanced_imaging_per_beneficiary": frec.get("advanced_imaging_per_beneficiary"),
            "dual_eligible_pct": frec.get("dual_eligible_pct"),
            "disabled_pct": frec.get("disabled_pct"),
            "average_risk_score": frec.get("average_available_risk_score")
        }

    return {"data": data, "sources": sources, "records": seg_records}
