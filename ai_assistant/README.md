# 🏥 Standalone AI Assistant Module for ACO Analytics & Value-Based Care

An enterprise-grade, read-only AI Assistant module designed for payer-facing **ACO Healthcare Analytics Command Centers**. Built with **Ollama** (for local LLM inference) and **Supabase/PostgreSQL** (for verified analytics and ML data retrieval), featuring dual integration via direct Python library import or standalone FastAPI REST microservice.

---

## 🌟 Key Architecture Principles

1. **LLM is NOT the Source of Truth**:
   The assistant never calculates or hallucinates metrics on the fly. All numerical values (savings rates, benchmark variances, quality scores, PMPM, utilization counts) are retrieved directly from verified Supabase / ML tables. The LLM's role is strictly to explain the retrieved facts clearly in natural language.
2. **Selective Table Search**:
   The system never searches all tables blindly. Questions are classified by intent and routed **strictly to the relevant Supabase tables**.
3. **Dual Integration Contract for Backend Developer**:
   - **Direct Python Import**: `from ai_assistant import ask`
   - **FastAPI HTTP Endpoint**: `POST http://localhost:8005/api/assistant/ask`
4. **Source Lineage & Grounded Auditability**:
   Every response includes a `sources` array listing the exact database tables used.
5. **Categorization Discipline**:
   Strictly distinguishes between **ACTUAL** historical data, **PREDICTED** ML outputs, **SIMULATED** scenarios, and **RECOMMENDATIONS**.

---

## 📁 Project Structure

```
chatbot/
│
├── ai_assistant/                      # Core Assistant Python Package
│   ├── __init__.py                    # Exports: ask, ACOAssistant, AssistantResponse, app
│   ├── api.py                         # Standalone FastAPI REST Microservice
│   ├── assistant.py                   # Central coordination pipeline
│   ├── config.py                      # Environment and configuration loader
│   ├── router.py                      # Intent classifier & selective table router
│   ├── context.py                     # Context resolver & conversation memory
│   ├── guardrails.py                  # Read-only guardrails & injection filters
│   ├── response_formatter.py          # Standard JSON contract formatter
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── ollama_client.py           # Ollama client with fallback grounding
│   │   └── prompts.py                 # Executive VBC system prompts
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── supabase_client.py         # Supabase connection manager
│   │   ├── dal.py                     # Data Access Layer functions
│   │   └── mock_data.py               # In-memory fallback mock dataset
│   │
│   └── tools/                         # Domain-specific controlled tools
│       ├── __init__.py
│       ├── financial_tools.py         # Queries fact_aco_performance & aco_financial_ml_training
│       ├── quality_tools.py           # Queries aco_analytics
│       ├── utilization_tools.py       # Queries aco_utilization_ml_features
│       ├── driver_tools.py            # Queries aco_analytics, utilization & segmentation features
│       ├── segmentation_tools.py      # Queries aco_segmentation_results & ml_features
│       ├── anomaly_tools.py           # Queries aco_anomalies, aco_alert, aco_risk_scores
│       ├── peer_tools.py              # Queries peer benchmarks & cohort targets
│       ├── provider_tools.py          # Queries bhavya_provider_aco_features_final
│       ├── service_tools.py           # Queries service_metrics
│       ├── whatif_tools.py            # Interprets What-if simulation results
│       └── recommendation_tools.py    # Synthesizes payer-level strategic recommendations
│
├── tests/                             # Automated Test Suite
│   ├── __init__.py
│   ├── sample_payloads.py             # Realistic input test cases across all domains
│   ├── test_tools.py                  # Data tool unit tests
│   ├── test_router.py                 # Selective table routing tests
│   ├── test_assistant.py              # End-to-end assistant tests
│   └── test_api.py                    # FastAPI endpoint tests
│
├── examples/                          # Quickstart & Integration Examples
│   ├── test_run.py                    # Batch and interactive CLI runner
│   ├── backend_integration.py         # Teammate Python integration snippet
│   ├── sample_curl_requests.bat       # Windows cURL test script
│   └── sample_curl_requests.sh        # Linux/macOS cURL test script
│
├── create_zip.py                      # Generates aco_ai_assistant_module.zip
├── requirements.txt                   # Project dependencies
├── .env.example                       # Environment variables template
├── .env                               # Live configuration file
└── README.md                          # Comprehensive documentation
```

---

## 🗄️ Supabase Schema & Selective Table Routing

| User Intent | Targeted Supabase Tables | Extracted Analytics |
| :--- | :--- | :--- |
| **FINANCIAL** | `fact_aco_performance`, `aco_financial_ml_training` | Benchmark, Expenditure, Gross Savings, Earned Savings, Savings %, PMPM |
| **QUALITY** | `aco_analytics`, `fact_aco_performance` | Quality Score, YoY Quality Change, Gap to 100, CAHPS Patient Experience (cahps_1 to 11) |
| **UTILIZATION** | `aco_utilization_ml_features`, `fact_aco_performance` | ED Visits, Inpatient Admissions, SNF Stays, CT/MRI Scans, E&M PCP/Specialist Visits |
| **PERFORMANCE_DRIVER** | `aco_analytics`, `aco_utilization_ml_features`, `aco_segmentation_ml_features` | Synthesizes financial, quality, and utilization drivers behind performance |
| **SEGMENTATION** | `aco_segmentation_results`, `aco_segmentation_ml_features` | Cluster ID, Performance Segment (High Performing / High Attention), Performance Index |
| **ANOMALY_RISK** | `aco_anomalies`, `aco_alert`, `aco_risk_scores`, `aco_anomaly_features` | Anomaly Score, Severity Level, Top Anomalous Metric, ML Risk Level, Alert Messages |
| **PEER_COMPARISON** | `aco_segmentation_results`, `fact_aco_performance` | Peer benchmarking, cohort averages, performance gap against top peer |
| **RECOMMENDATION** | `aco_alert`, `aco_risk_scores`, `aco_segmentation_results` | Payer-level strategic recommendations derived from risk alerts & quality gaps |
| **PROVIDER_ANALYTICS** | `bhavya_provider_aco_features_final` | Provider NPI, Provider Segment, Cost/Utilization Scores, Specialty, Payments |
| **SERVICE_ANALYTICS** | `service_metrics` | HCPCS Codes, Service Categories, Payment per Beneficiary, Volume |
| **WHAT_IF** | *(Simulator Engine)* | Parses simulation context and clearly explains projected impacts |
| **DASHBOARD_OVERVIEW** | Active Page Context + `fact_aco_performance` + `aco_analytics` | Summarizes the active dashboard page/section for executive review |

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Verify your `.env` file contains your credentials:
```ini
SUPABASE_URL=https://wuazaratvveeemdpeait.supabase.co
SUPABASE_KEY=your-supabase-key
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
ASSISTANT_PORT=8005
```

### 3. Ensure Ollama is Running
In a separate terminal, verify Ollama is active with your model:
```bash
ollama run llama3.2
```

---

## 🧪 Running Tests & CLI Runner

### Run Complete Automated Test Suite (Pytest)
```bash
pytest
```

### Run Batch Test Runner (Tests all sample dashboard queries)
```bash
python examples/test_run.py
```

### Run Interactive Terminal Chat
```bash
python examples/test_run.py --interactive
```

---

## 🔌 Backend Integration Guide (For Teammates)

### Option A: Direct Python Package Import (Recommended)
Your backend teammate can directly import and call the assistant function:

```python
from ai_assistant import ask

result = ask(
    question="Why is this ACO classified as High Performing?",
    context={
        "aco_id": "A1001",
        "year": 2021,
        "page": "ACO Explorer",
        "section": "Performance Drivers",
        "selected_metric": "Savings"
    }
)

# Send result directly to React frontend
print(result["answer"])
print(result["sources"])
```

### Option B: Standalone FastAPI Microservice
Start the assistant as an independent API server:
```bash
uvicorn ai_assistant.api:app --host 0.0.0.0 --port 8005
```

Interactive API documentation available at: `http://localhost:8005/docs`

#### Sample HTTP Request (cURL):
```bash
curl -X POST http://localhost:8005/api/assistant/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was ACO A1001 savings rate in 2021?",
    "context": {
      "aco_id": "A1001",
      "year": 2021,
      "page": "ACO Explorer"
    }
  }'
```

---

## 📦 Standard JSON Response Contract

Every response adheres to this guaranteed schema:

```json
{
  "answer": "In 2021, **Palm Beach Accountable Care Organization (A1001)** achieved strong performance:\n- **Gross Savings**: $84,231,357 (Savings Rate: **7.17%**)\n- **Quality Score**: **100.0%**\n- **Assigned Beneficiaries**: 89,403\n- **Risk Model**: Two-Sided",
  "intent": "FINANCIAL",
  "status": "success",
  "context_used": {
    "aco_id": "A1001",
    "year": 2021,
    "page": "ACO Explorer",
    "section": "Financial Performance",
    "selected_metric": null,
    "compare_with_aco_id": null
  },
  "sources": [
    "fact_aco_performance"
  ],
  "data": {
    "aco_id": "A1001",
    "aco_name": "Palm Beach Accountable Care Organization",
    "state": "FL",
    "year": 2021,
    "total_benchmark": 1084231357,
    "total_expenditure": 1000000000,
    "gross_savings_loss": 84231357,
    "savings_rate_pct": 7.17,
    "quality_score": 100.0,
    "assigned_beneficiaries": 89403
  },
  "warnings": []
}
```

---

## 📦 Generating Project ZIP File

To package the entire codebase into a clean zip archive for distribution:
```bash
python create_zip.py
```
This generates `aco_ai_assistant_module.zip` in the root folder.
