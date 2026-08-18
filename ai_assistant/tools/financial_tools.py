"""
Financial Data Extraction Tools.
Sources: fact_aco_performance, aco_financial_ml_training
"""

from typing import Any, Dict, List, Optional
from ai_assistant.data import dal

def get_financial_metrics(aco_id: str, year: Optional[int] = None) -> Dict[str, Any]:
    """
    Extracts verified financial metrics (Expenditure, Benchmark, Savings/Loss, PMPM).
    Priority: fact_aco_performance -> aco_financial_ml_training
    """
    sources: List[str] = []
    data: Dict[str, Any] = {}

    # 1. Query fact_aco_performance
    fact_records = dal.get_fact_aco_performance(aco_id, year)
    if fact_records:
        sources.append("fact_aco_performance")
        record = fact_records[0]
        data = {
            "aco_id": record.get("ACO_ID"),
            "aco_name": record.get("ACO_Name"),
            "state": record.get("ACO_State"),
            "year": record.get("performance_year"),
            "total_benchmark": record.get("ABtotBnchmk"),
            "total_expenditure": record.get("ABtotExp"),
            "gross_savings_loss": record.get("GenSaveLoss"),
            "earned_savings_loss": record.get("EarnSaveLoss"),
            "savings_rate_pct": round(record.get("Sav_rate", 0) * 100, 2) if record.get("Sav_rate") is not None else None,
            "quality_score": record.get("QualScore"),
            "assigned_beneficiaries": record.get("N_AB"),
            "per_capita_expenditure": record.get("Per_Capita_Exp_TOTAL_PY"),
            "risk_model": record.get("Risk_Model"),
            "revenue_category": record.get("Rev_Exp_Cat")
        }
        return {"data": data, "sources": sources, "records": fact_records}

    # 2. Query aco_financial_ml_training fallback
    ml_records = dal.get_aco_financial_ml_training(aco_id, year)
    if ml_records:
        sources.append("aco_financial_ml_training")
        record = ml_records[0]
        data = {
            "aco_id": record.get("ACO_ID"),
            "year": record.get("target_year") or record.get("feature_year"),
            "total_benchmark": record.get("ABtotBnchmk"),
            "total_expenditure": record.get("ABtotExp"),
            "gross_savings_loss": record.get("GenSaveLoss"),
            "earned_savings_loss": record.get("EarnSaveLoss"),
            "savings_rate_pct": round(record.get("SavingsLossPct", 0), 2) if record.get("SavingsLossPct") is not None else None,
            "expenditure_variance_pct": record.get("ExpenditureVariancePct"),
            "pmpm": record.get("PMPM"),
            "benchmark_pmpm": record.get("BenchmarkPMPM"),
            "financial_gap": record.get("FinancialGap"),
            "quality_score": record.get("QualScore"),
            "assigned_beneficiaries": record.get("N_AB"),
            "risk_model": record.get("Risk_Model")
        }
        return {"data": data, "sources": sources, "records": ml_records}

    return {"data": {}, "sources": sources, "records": []}

def get_financial_history(aco_id: str) -> Dict[str, Any]:
    """Extracts longitudinal multi-year financial history for trend analysis."""
    records = dal.get_fact_aco_performance(aco_id)
    sources = ["fact_aco_performance"]
    
    if not records:
        records = dal.get_aco_financial_ml_training(aco_id)
        sources = ["aco_financial_ml_training"]

    history = []
    for r in records:
        yr = r.get("performance_year") or r.get("target_year") or r.get("feature_year")
        sav = r.get("Sav_rate") or r.get("SavingsLossPct")
        if sav is not None and sav < 1.0: # If ratio, convert to percentage
            sav = sav * 100
        history.append({
            "year": yr,
            "expenditure": r.get("ABtotExp"),
            "benchmark": r.get("ABtotBnchmk"),
            "savings_loss": r.get("GenSaveLoss"),
            "savings_rate_pct": round(sav, 2) if sav is not None else None,
            "quality_score": r.get("QualScore")
        })

    history = sorted(history, key=lambda x: x["year"] if x["year"] else 0)
    return {"history": history, "sources": sources}
