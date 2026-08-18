"""
Main AI Assistant Pipeline & Coordination Engine.
Connects Context Resolution -> Intent Routing -> Selective Tool Execution -> Ollama LLM Grounding -> Response Formatting.
"""

from typing import Any, Dict, List, Optional
from ai_assistant.config import settings
from ai_assistant.context import resolve_context
from ai_assistant.router import classify_intent, IntentType
from ai_assistant.guardrails import sanitize_and_validate_input, evaluate_data_presence
from ai_assistant.response_formatter import format_response
from ai_assistant.llm.prompts import build_prompt, SYSTEM_PROMPT
from ai_assistant.llm.ollama_client import generate_ollama_response
from ai_assistant import tools

class ACOAssistant:
    """Core AI Assistant Orchestrator for Value-Based Care Analytics."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.OLLAMA_MODEL

    def ask(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for processing a user query.
        Returns a standardized, human-readable JSON response.
        """
        # Step 1: Input Validation
        is_valid, error_msg = sanitize_and_validate_input(question)
        if not is_valid:
            return format_response(
                answer=f"Error: {error_msg}",
                intent="INVALID",
                status="error",
                warnings=[error_msg or "Invalid request"]
            )

        # Step 2: Context Resolution
        resolved_ctx = resolve_context(question, context, session_id)
        aco_id = resolved_ctx.get("aco_id")
        year = resolved_ctx.get("year")
        compare_with_aco_id = resolved_ctx.get("compare_with_aco_id")

        # Step 3: Intent Classification & Selective Table Routing
        intent, target_tables = classify_intent(question, resolved_ctx)

        # Step 4: Selective Tool Execution
        tool_result = self._execute_targeted_tool(intent, aco_id, year, compare_with_aco_id, resolved_ctx)
        extracted_data = tool_result.get("data", {})
        used_sources = tool_result.get("sources", target_tables)

        # Step 5: Data Presence Check & Warnings
        warnings = evaluate_data_presence(aco_id, year, extracted_data)

        # If no data found for specified ACO
        if aco_id and not extracted_data:
            return format_response(
                answer=f"I couldn't find records for **{aco_id}** in the available dataset.",
                intent=intent,
                status="success",
                context_used=resolved_ctx,
                sources=used_sources,
                data={},
                warnings=[f"ACO {aco_id} not found in database"]
            )

        # Step 6: Grounded Prompt Assembly & Ollama Inference
        prompt = build_prompt(
            question=question,
            intent=intent,
            context=resolved_ctx,
            retrieved_data=extracted_data,
            sources=used_sources
        )

        llm_answer = generate_ollama_response(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            model=self.model_name
        )

        # Step 7: Standardized JSON Output
        return format_response(
            answer=llm_answer,
            intent=intent,
            status="success",
            context_used=resolved_ctx,
            sources=used_sources,
            data=extracted_data,
            warnings=warnings
        )

    def _execute_targeted_tool(
        self,
        intent: str,
        aco_id: Optional[str],
        year: Optional[int],
        compare_with_aco_id: Optional[str],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Dispatches query only to the relevant tool based on classified intent."""
        # Default fallback ACO if unspecified
        target_aco = aco_id or "A1001"

        if intent == IntentType.FINANCIAL:
            return tools.get_financial_metrics(target_aco, year)

        elif intent == IntentType.QUALITY:
            return tools.get_quality_metrics(target_aco, year)

        elif intent == IntentType.UTILIZATION:
            return tools.get_utilization_metrics(target_aco, year)

        elif intent == IntentType.HISTORICAL_TREND:
            fin_hist = tools.get_financial_history(target_aco)
            qual_hist = tools.get_quality_history(target_aco)
            return {
                "data": {
                    "financial_trend": fin_hist.get("history", []),
                    "quality_trend": qual_hist.get("history", [])
                },
                "sources": list(set(fin_hist.get("sources", []) + qual_hist.get("sources", [])))
            }

        elif intent == IntentType.PERFORMANCE_DRIVER:
            return tools.get_performance_drivers(target_aco, year)

        elif intent == IntentType.SEGMENTATION:
            return tools.get_segmentation_results(target_aco, year)

        elif intent == IntentType.ANOMALY_RISK:
            return tools.get_anomaly_and_risk(target_aco, year)

        elif intent == IntentType.PEER_COMPARISON:
            return tools.get_peer_comparison(target_aco, year, compare_with_aco_id)

        elif intent == IntentType.RECOMMENDATION:
            return tools.get_payer_recommendations(target_aco, year)

        elif intent == IntentType.PROVIDER_ANALYTICS:
            return tools.get_provider_details(aco_id=aco_id, year=year)

        elif intent == IntentType.SERVICE_ANALYTICS:
            return tools.get_service_analytics(target_aco, year)

        elif intent == IntentType.WHAT_IF:
            return tools.interpret_whatif_scenario(context)

        elif intent == IntentType.DASHBOARD_OVERVIEW:
            fin = tools.get_financial_metrics(target_aco, year)
            qual = tools.get_quality_metrics(target_aco, year)
            seg = tools.get_segmentation_results(target_aco, year)
            return {
                "data": {
                    "financial_summary": fin.get("data"),
                    "quality_summary": qual.get("data"),
                    "segmentation": seg.get("data")
                },
                "sources": list(set(fin.get("sources", []) + qual.get("sources", []) + seg.get("sources", [])))
            }

        else: # GENERAL_DATA fallback
            return tools.get_financial_metrics(target_aco, year)

# Top-level standalone function for direct import: `from ai_assistant import ask`
_default_assistant = ACOAssistant()

def ask(
    question: str,
    context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main standalone function.
    Usage:
        result = ask(question="What was A1001's savings in 2021?")
    """
    return _default_assistant.ask(question=question, context=context, session_id=session_id)
