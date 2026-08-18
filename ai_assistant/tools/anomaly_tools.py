"""
Anomaly & Risk Data Extraction Tools.
Sources: aco_anomalies, aco_alert, aco_risk_scores, aco_anomaly_features
"""

from typing import Any, Dict, List, Optional
from ai_assistant.data import dal

def get_anomaly_and_risk(aco_id: str, year: Optional[int] = None) -> Dict[str, Any]:
    """
    Extracts anomaly flags, ML risk scores, top anomalous metrics, and alert messages.
    """
    sources: List[str] = []
    data: Dict[str, Any] = {
        "aco_id": aco_id,
        "year": year,
        "anomalies": [],
        "alerts": [],
        "risk_scores": [],
        "feature_changes": {}
    }

    # 1. Anomaly alerts
    alerts = dal.get_aco_alerts(aco_id, year)
    if alerts:
        sources.append("aco_alert")
        data["alerts"] = [
            {
                "alert_type": a.get("alert_type"),
                "severity": a.get("severity"),
                "message": a.get("message"),
                "year": a.get("performance_year")
            } for a in alerts
        ]

    # 2. Detected Anomalies
    anomalies = dal.get_aco_anomalies(aco_id, year)
    if anomalies:
        sources.append("aco_anomalies")
        data["anomalies"] = [
            {
                "anomaly_score": an.get("anomaly_score"),
                "top_metric": an.get("top_anomalous_metric"),
                "severity": an.get("severity"),
                "year": an.get("performance_year")
            } for an in anomalies
        ]

    # 3. Risk Scores
    risk_scores = dal.get_aco_risk_scores(aco_id, year)
    if risk_scores:
        sources.append("aco_risk_scores")
        data["risk_scores"] = [
            {
                "ml_anomaly_score": r.get("ml_anomaly_score"),
                "risk_level": r.get("risk_level"),
                "is_anomaly": r.get("is_anomaly"),
                "year": r.get("performance_year")
            } for r in risk_scores
        ]

    # 4. Anomaly Features
    features = dal.get_aco_anomaly_features(aco_id, year)
    if features:
        sources.append("aco_anomaly_features")
        f = features[0]
        data["feature_changes"] = {
            "savings_yoy_change_pct": f.get("savings_yoy_change_pct"),
            "expenditure_variance_pct": f.get("expenditure_variance_pct"),
            "quality_change_yoy": f.get("quality_change_yoy"),
            "ed_utilization_change_yoy": f.get("ed_utilization_change_yoy"),
            "admission_change_yoy": f.get("admission_change_yoy"),
            "em_utilization_change_yoy": f.get("em_utilization_change_yoy"),
            "advanced_imaging_change_yoy": f.get("advanced_imaging_change_yoy")
        }

    return {"data": data, "sources": sources}
