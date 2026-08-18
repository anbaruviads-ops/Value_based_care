"""
Utilization Data Extraction Tools.
Sources: aco_utilization_ml_features, fact_aco_performance
"""

from typing import Any, Dict, List, Optional
from ai_assistant.data import dal

def get_utilization_metrics(aco_id: str, year: Optional[int] = None) -> Dict[str, Any]:
    """
    Extracts utilization metrics (ED visits, hospital admissions, imaging, E&M visits, SNF).
    """
    sources: List[str] = []
    data: Dict[str, Any] = {}

    util_records = dal.get_aco_utilization_features(aco_id, year)
    if util_records:
        sources.append("aco_utilization_ml_features")
        rec = util_records[0]
        data = {
            "aco_id": rec.get("ACO_ID"),
            "year": rec.get("performance_year"),
            "beneficiaries": rec.get("N_AB"),
            "ed_visits": rec.get("P_EDV_Vis"),
            "ed_visits_hospital": rec.get("P_EDV_Vis_HOSP"),
            "admissions": rec.get("ADM"),
            "short_term_admissions": rec.get("ADM_S_Trm"),
            "snf_admissions": rec.get("P_SNF_ADM"),
            "ct_scans": rec.get("P_CT_VIS"),
            "mri_scans": rec.get("P_MRI_VIS"),
            "em_visits_total": rec.get("P_EM_Total"),
            "em_pcp_visits": rec.get("P_EM_PCP_Vis"),
            "em_specialist_visits": rec.get("P_EM_SP_Vis"),
            "emergency_rate_per_1000": rec.get("emergency_utilization_rate"),
            "admission_rate_per_1000": rec.get("admission_rate_per_1000"),
            "ed_utilization_change_yoy": rec.get("ed_utilization_change_yoy"),
            "admission_change_yoy": rec.get("admission_change_yoy"),
            "em_utilization_change_yoy": rec.get("em_utilization_change_yoy"),
            "advanced_imaging_change_yoy": rec.get("advanced_imaging_change_yoy"),
            "utilization_score": rec.get("utilization_score"),
            "utilization_category": rec.get("utilization_category"),
            "high_utilization_flag": rec.get("high_utilization_flag"),
            "low_utilization_flag": rec.get("low_utilization_flag")
        }
        return {"data": data, "sources": sources, "records": util_records}

    # Fallback to fact_aco_performance
    fact_records = dal.get_fact_aco_performance(aco_id, year)
    if fact_records:
        sources.append("fact_aco_performance")
        rec = fact_records[0]
        data = {
            "aco_id": rec.get("ACO_ID"),
            "year": rec.get("performance_year"),
            "ed_visits": rec.get("P_EDV_Vis"),
            "admissions": rec.get("ADM"),
            "em_visits_total": rec.get("P_EM_Total"),
            "ct_scans": rec.get("P_CT_VIS"),
            "mri_scans": rec.get("P_MRI_VIS"),
            "snf_admissions": rec.get("P_SNF_ADM")
        }
        return {"data": data, "sources": sources, "records": fact_records}

    return {"data": {}, "sources": sources, "records": []}
