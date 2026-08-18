"""
Provider Data Extraction Tools.
Sources: bhavya_provider_aco_features_final
"""

from typing import Any, Dict, List, Optional
from ai_assistant.data import dal

def get_provider_details(aco_id: Optional[str] = None, npi: Optional[int] = None, year: Optional[int] = None) -> Dict[str, Any]:
    """
    Extracts provider-level performance, utilization, cost scores, and service metrics.
    """
    sources = ["bhavya_provider_aco_features_final"]
    records = dal.get_provider_features(aco_id=aco_id, npi=npi, year=year, limit=10)

    providers = []
    for r in records:
        providers.append({
            "npi": r.get("rndrng_npi"),
            "provider_name": f"{r.get('rndrng_prvdr_first_name', '')} {r.get('rndrng_prvdr_last_org_name', '')}".strip(),
            "specialty": r.get("rndrng_prvdr_type"),
            "city_state": f"{r.get('rndrng_prvdr_city', '')}, {r.get('rndrng_prvdr_state_abrvtn', '')}",
            "aco_id": r.get("aco_id"),
            "total_beneficiaries": r.get("tot_benes"),
            "total_services": r.get("tot_srvcs"),
            "medicare_payment": r.get("tot_mdcr_pymt_amt"),
            "services_per_beneficiary": r.get("services_per_beneficiary"),
            "payment_per_beneficiary": r.get("payment_per_beneficiary"),
            "provider_segment": r.get("provider_segment"),
            "high_cost_flag": r.get("high_cost_flag"),
            "high_utilization_flag": r.get("high_utilization_flag")
        })

    return {"data": {"providers": providers, "count": len(providers)}, "sources": sources}
