"""
Payer Strategic Recommendation Tools.
Sources: aco_alert, aco_risk_scores, aco_segmentation_results, aco_anomalies
"""

from typing import Any, Dict, List, Optional
from ai_assistant.data import dal

def get_payer_recommendations(aco_id: str, year: Optional[int] = None) -> Dict[str, Any]:
    """
    Synthesizes stored ML outputs, risk alerts, and quality gaps into strategic payer recommendations.
    Does not invent clinical medical interventions; focuses strictly on VBC payer management.
    """
    sources: List[str] = []
    recommendations: List[Dict[str, str]] = []

    # 1. Check alerts & anomalies
    alerts = dal.get_aco_alerts(aco_id, year)
    if alerts:
        sources.append("aco_alert")
        for a in alerts:
            recommendations.append({
                "category": "Utilization Alert",
                "finding": a.get("message", ""),
                "action": f"Initiate utilization review regarding {a.get('alert_type', 'flagged metric')} (Severity: {a.get('severity')})."
            })

    # 2. Check quality gaps
    quality = dal.get_aco_analytics(aco_id, year)
    if quality:
        sources.append("aco_analytics")
        q = quality[0]
        gap = q.get("quality_gap_to_100", 0)
        if gap and gap > 5.0:
            recommendations.append({
                "category": "Quality Performance",
                "finding": f"Current quality score has a {gap:.1f} point gap to 100.",
                "action": "Focus on CAHPS patient experience and preventive screening compliance measures."
            })

    # 3. Check segmentation
    seg = dal.get_aco_segmentation(aco_id, year)
    if seg:
        sources.append("aco_segmentation_results")
        s = seg[0]
        segment_name = s.get("performance_segment", "")
        if "Attention" in segment_name or "High Risk" in segment_name:
            recommendations.append({
                "category": "Contract & Portfolio Risk",
                "finding": f"ACO classified as '{segment_name}'.",
                "action": "Schedule quarterly contract review and audit provider outlier variations."
            })

    if not recommendations:
        recommendations.append({
            "category": "Performance Maintenance",
            "finding": "ACO is performing within expected benchmarks with no active critical anomaly alerts.",
            "action": "Maintain routine longitudinal monitoring and peer benchmark tracking."
        })

    return {"data": {"recommendations": recommendations}, "sources": sources}
