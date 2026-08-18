"""
Universal LLM Client supporting Local Ollama, Remote Ollama, Cloud Fallback (Groq / OpenAI),
and Structured Synthesis.
"""

import json
import logging
import requests
from typing import Any, Dict, List, Optional
from ai_assistant.config import settings
from ai_assistant.llm.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

def generate_ollama_response(
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    model: Optional[str] = None,
    timeout: Optional[int] = None
) -> str:
    """
    Orchestrates LLM inference with tiered failover:
    1. Local/Remote Ollama (Default: llama3.2 on http://localhost:11434)
    2. Cloud LLM (Groq API - Free fast Llama 3.2 inference if GROQ_API_KEY is present)
    3. OpenAI API (if OPENAI_API_KEY is present)
    4. Deterministic Factual Grounding (Zero-crash fallback if completely offline)
    """
    # Tier 1: Try Ollama
    ollama_text = _call_ollama(prompt, system_prompt, model, timeout)
    if ollama_text:
        return ollama_text

    # Tier 2: Try Groq Cloud (Free Llama 3.2 inference for remote teammates)
    if settings.GROQ_API_KEY:
        groq_text = _call_groq_api(prompt, system_prompt)
        if groq_text:
            return groq_text

    # Tier 3: Try OpenAI (if configured)
    if settings.OPENAI_API_KEY:
        openai_text = _call_openai_api(prompt, system_prompt)
        if openai_text:
            return openai_text

    # Tier 4: Factual Fallback Synthesis
    return _fallback_grounded_synthesis(prompt)

def _call_ollama(
    prompt: str,
    system_prompt: str,
    model: Optional[str],
    timeout: Optional[int]
) -> Optional[str]:
    """Sends generation request to Ollama endpoint."""
    target_model = model or settings.OLLAMA_MODEL
    api_url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    req_timeout = timeout or settings.OLLAMA_TIMEOUT_SECONDS

    payload = {
        "model": target_model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9
        }
    }

    try:
        response = requests.post(api_url, json=payload, timeout=req_timeout)
        if response.status_code == 200:
            result = response.json()
            llm_text = result.get("response", "").strip()
            if llm_text:
                return llm_text
    except Exception as e:
        logger.debug(f"Ollama call failed or offline: {e}")

    return None

def _call_groq_api(prompt: str, system_prompt: str) -> Optional[str]:
    """Calls Groq Cloud API for free fast Llama3.2/3.1 inference."""
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"Groq API call failed: {e}")
    return None

def _call_openai_api(prompt: str, system_prompt: str) -> Optional[str]:
    """Calls OpenAI API if configured."""
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"OpenAI API call failed: {e}")
    return None

def _fallback_grounded_synthesis(prompt: str) -> str:
    """
    Generates a clean, executive bulleted synthesis directly from the grounded context
    when no LLM server is accessible.
    """
    lines = prompt.splitlines()
    data_section = False
    data_str = ""
    for line in lines:
        if "VERIFIED DATA FROM SUPABASE" in line or "VERIFIED DATA EXTRACTED" in line:
            data_section = True
            continue
        if data_section:
            if "Please generate" in line:
                break
            data_str += line + "\n"

    try:
        import ast
        data = ast.literal_eval(data_str.strip())
    except Exception:
        data = {}

    if not data:
        return "No matching analytical records found in the database for the specified parameters."

    output_parts = ["**Verified ACO Analytics Summary:**\n"]

    if "total_expenditure" in data:
        exp = f"${data.get('total_expenditure', 0):,}" if isinstance(data.get('total_expenditure'), (int, float)) else data.get('total_expenditure')
        bnch = f"${data.get('total_benchmark', 0):,}" if isinstance(data.get('total_benchmark'), (int, float)) else data.get('total_benchmark')
        sav = f"${data.get('gross_savings_loss', 0):,}" if isinstance(data.get('gross_savings_loss'), (int, float)) else data.get('gross_savings_loss')
        output_parts.append(f"- **Total Expenditure**: {exp}")
        output_parts.append(f"- **Total Benchmark**: {bnch}")
        output_parts.append(f"- **Gross Savings/Loss**: {sav} (Rate: **{data.get('savings_rate_pct', 'N/A')}%**)")
        if data.get("quality_score") is not None:
            output_parts.append(f"- **Quality Score**: **{data.get('quality_score')}%**")
        if data.get("assigned_beneficiaries"):
            output_parts.append(f"- **Assigned Beneficiaries**: {data.get('assigned_beneficiaries'):,}")

    elif "quality_score" in data:
        output_parts.append(f"- **Quality Score**: **{data.get('quality_score')}%** (Category: **{data.get('quality_category', 'Standard')}**)")
        if data.get("quality_gap_to_100") is not None:
            output_parts.append(f"- **Quality Gap to 100**: {data.get('quality_gap_to_100')} points")
        if data.get("cahps_scores"):
            cahps = data.get("cahps_scores", {})
            output_parts.append(f"- **Patient Experience (CAHPS)**: Provider Comm: {cahps.get('provider_communication', 'N/A')}, Timely Care: {cahps.get('getting_timely_care', 'N/A')}")

    elif "ed_visits" in data:
        output_parts.append(f"- **ED Visits**: {data.get('ed_visits')} (Rate: **{data.get('emergency_rate_per_1000', 'N/A')} per 1,000**)")
        output_parts.append(f"- **Hospital Admissions**: {data.get('admissions')} (Rate: **{data.get('admission_rate_per_1000', 'N/A')} per 1,000**)")
        output_parts.append(f"- **Utilization Category**: **{data.get('utilization_category', 'Standard')}** (Score: {data.get('utilization_score', 'N/A')})")

    elif "performance_segment" in data:
        output_parts.append(f"- **ML Performance Segment**: **{data.get('performance_segment')}** (Cluster: {data.get('cluster_id')})")
        output_parts.append(f"- **Performance Index**: **{data.get('performance_index', 'N/A')}**")
        if data.get("savings_loss_pct") is not None:
            output_parts.append(f"- **Savings/Loss %**: {data.get('savings_loss_pct')}%")
        if data.get("quality_score") is not None:
            output_parts.append(f"- **Quality Score**: {data.get('quality_score')}%")

    elif "alerts" in data or "anomalies" in data:
        alerts = data.get("alerts", [])
        anomalies = data.get("anomalies", [])
        if alerts:
            for a in alerts:
                output_parts.append(f"- ⚠️ **Alert ({a.get('severity', 'HIGH')})**: {a.get('message')}")
        if anomalies:
            for an in anomalies:
                output_parts.append(f"- **Top Anomalous Metric**: {an.get('top_metric')} (Score: {an.get('anomaly_score'):.2f}, Severity: {an.get('severity')})")

    elif "type" in data and data.get("type") == "SIMULATED_SCENARIO":
        scenario = data.get("scenario_input", {})
        result = data.get("projected_result", {})
        output_parts.append(f"*(SIMULATED SCENARIO)*: Applying **{scenario.get('metric', 'parameter')}** change of **{scenario.get('change', '0%')}**:")
        for k, v in result.items():
            output_parts.append(f"- **{k.replace('_', ' ').title()}**: **{v}**")
        output_parts.append(f"\n*{data.get('disclaimer')}*")

    elif "recommendations" in data:
        for r in data.get("recommendations", []):
            output_parts.append(f"- **{r.get('category')}**: {r.get('action')}")

    elif "providers" in data:
        output_parts.append(f"Found **{data.get('count', 0)}** provider records matching criteria:")
        for p in data.get("providers", [])[:3]:
            output_parts.append(f"- **Dr. {p.get('provider_name')}** ({p.get('specialty')}): Segment **{p.get('provider_segment')}**, Payment/Bene: **${p.get('payment_per_beneficiary')}**")

    else:
        for k, v in list(data.items())[:6]:
            output_parts.append(f"- **{k.replace('_', ' ').title()}**: {v}")

    return "\n".join(output_parts)
