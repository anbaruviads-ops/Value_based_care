"""
What-If Simulator Context & Projection Formatter.
Interprets controlled simulation parameters and outcomes.
Labels output strictly as SIMULATED / PROJECTED.
"""

from typing import Any, Dict

def interpret_whatif_scenario(scenario_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses and verifies What-if simulator input and output.
    """
    scenario = scenario_context.get("scenario", {})
    result = scenario_context.get("result", {})

    return {
        "data": {
            "type": "SIMULATED_SCENARIO",
            "scenario_input": scenario,
            "projected_result": result,
            "disclaimer": "This is a simulated projection based on the What-if engine and does not represent actual historical financial or clinical results."
        },
        "sources": ["what_if_simulator_engine"]
    }
