"""
Peer Comparison & Benchmarking Tools.
Sources: aco_segmentation_results, fact_aco_performance
"""

from typing import Any, Dict, List, Optional
from ai_assistant.data import dal

def get_peer_comparison(aco_id: str, year: Optional[int] = None, compare_with_aco_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Compares an ACO with peers in the same segment or against a specific target peer ACO.
    """
    sources: List[str] = []
    
    # Get current ACO segment
    current_seg = dal.get_aco_segmentation(aco_id, year)
    segment_name = current_seg[0].get("performance_segment") if current_seg else None
    
    if current_seg:
        sources.append("aco_segmentation_results")

    # If specific peer requested
    if compare_with_aco_id:
        target_seg = dal.get_aco_segmentation(compare_with_aco_id, year)
        return {
            "data": {
                "aco_1": current_seg[0] if current_seg else {"ACO_ID": aco_id},
                "aco_2": target_seg[0] if target_seg else {"ACO_ID": compare_with_aco_id}
            },
            "sources": sources
        }

    # Otherwise find peers in the same segment
    peers = dal.get_all_segmented_acos(segment=segment_name, year=year, limit=5)
    return {
        "data": {
            "target_aco": current_seg[0] if current_seg else {"ACO_ID": aco_id},
            "segment": segment_name,
            "peers": [p for p in peers if p.get("ACO_ID") != aco_id]
        },
        "sources": sources
    }
