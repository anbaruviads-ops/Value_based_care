"""
Sample Question & Dashboard Context Payloads.
Demonstrates realistic frontend/backend payloads across all ACO analytics domains.
"""

SAMPLE_PAYLOADS = [
    {
        "id": "financial_direct",
        "name": "Direct Financial Query",
        "payload": {
            "question": "What was ACO A1001's expenditure and savings in 2021?",
            "context": {
                "aco_id": "A1001",
                "year": 2021,
                "page": "ACO Explorer",
                "section": "Financial Performance"
            }
        }
    },
    {
        "id": "quality_direct",
        "name": "Quality Score & CAHPS Query",
        "payload": {
            "question": "What is the quality score and patient experience rating for A1001 in 2021?",
            "context": {
                "aco_id": "A1001",
                "year": 2021,
                "page": "ACO Explorer",
                "section": "Quality Metrics"
            }
        }
    },
    {
        "id": "utilization_direct",
        "name": "ED & Hospital Admission Utilization",
        "payload": {
            "question": "What was the emergency department utilization rate for A1001 in 2021?",
            "context": {
                "aco_id": "A1001",
                "year": 2021,
                "page": "ACO Explorer",
                "section": "Utilization"
            }
        }
    },
    {
        "id": "historical_trend",
        "name": "Longitudinal Performance Trend",
        "payload": {
            "question": "How did ACO A1001 perform financially and in quality over time?",
            "context": {
                "aco_id": "A1001",
                "page": "ACO Explorer",
                "section": "Historical Trends"
            }
        }
    },
    {
        "id": "performance_drivers",
        "name": "Performance Driver Analysis",
        "payload": {
            "question": "What are the main performance drivers for A1001?",
            "context": {
                "aco_id": "A1001",
                "year": 2021,
                "page": "ACO Explorer",
                "section": "Performance Drivers"
            }
        }
    },
    {
        "id": "segmentation_cluster",
        "name": "ML Segmentation & Classification",
        "payload": {
            "question": "Which segment is ACO A1001 classified into and why?",
            "context": {
                "aco_id": "A1001",
                "year": 2021,
                "page": "Portfolio Overview",
                "section": "Segmentation"
            }
        }
    },
    {
        "id": "anomaly_alert",
        "name": "Anomaly Detection & Severity Alert",
        "payload": {
            "question": "Are there any high-severity utilization anomalies or alerts flagged for A1199?",
            "context": {
                "aco_id": "A1199",
                "year": 2023,
                "page": "Executive Alerts",
                "section": "Anomaly Detection"
            }
        }
    },
    {
        "id": "peer_comparison",
        "name": "Peer Benchmarking",
        "payload": {
            "question": "How does ACO A1001 compare with other peer ACOs in its segment?",
            "context": {
                "aco_id": "A1001",
                "year": 2021,
                "page": "Peer Comparison",
                "section": "Cohort Analysis"
            }
        }
    },
    {
        "id": "provider_drilldown",
        "name": "Provider Level Drill-down",
        "payload": {
            "question": "Show provider cost and utilization variations for ACO_005.",
            "context": {
                "aco_id": "ACO_005",
                "year": 2024,
                "page": "Provider Drilldown",
                "section": "Provider Network"
            }
        }
    },
    {
        "id": "service_metrics",
        "name": "Service Category & HCPCS Code Analytics",
        "payload": {
            "question": "Which HCPCS service categories had highest utilization for A1001 in 2021?",
            "context": {
                "aco_id": "A1001",
                "year": 2021,
                "page": "Service Analytics",
                "section": "HCPCS Breakdown"
            }
        }
    },
    {
        "id": "whatif_simulation",
        "name": "What-If Simulator Scenario Explanation",
        "payload": {
            "question": "What is the projected financial impact under this simulation scenario?",
            "context": {
                "aco_id": "A1001",
                "scenario": {
                    "metric": "ER utilization",
                    "change": "-5%"
                },
                "result": {
                    "projected_expenditure_change": "-2.1%",
                    "estimated_net_savings_increase": "$1.8M"
                }
            }
        }
    },
    {
        "id": "dashboard_context_metric",
        "name": "Concise Question with Active Dashboard Context",
        "payload": {
            "question": "Why is this metric red?",
            "context": {
                "aco_id": "A1001",
                "year": 2021,
                "page": "ACO Explorer",
                "section": "Performance Drivers",
                "selected_metric": "Utilization"
            }
        }
    },
    {
        "id": "missing_data_aco",
        "name": "Nonexistent ACO (Hallucination Test)",
        "payload": {
            "question": "What was the savings percentage for nonexistent ACO-99999?",
            "context": {
                "aco_id": "A99999",
                "year": 2024
            }
        }
    }
]
