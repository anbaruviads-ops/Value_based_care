"""
In-Memory Mock/Fallback Database matching the exact Supabase Schema & Sample Data.
Used when Supabase is unreachable or during offline development & unit testing.
"""

FACT_ACO_PERFORMANCE = [
    {
        "ACO_ID": "A1001",
        "ACO_Name": "Palm Beach Accountable Care Organization",
        "ACO_State": "FL",
        "performance_year": 2020,
        "Agree_Type": "Renewal",
        "Agreement_Period_Num": 3,
        "Risk_Model": "Two-Sided",
        "N_AB": 80864,
        "MinSavPerc": 0.02,
        "BnchmkMinExp": 72427998,
        "GenSaveLoss": 72427998,
        "EarnSaveLoss": 54320998,
        "QualScore": 100.0,
        "ABtotBnchmk": 956864945,
        "ABtotExp": 884436947,
        "FinalShareRate": 0.75,
        "RevLossLimit": 0.0,
        "Rev_Exp_Cat": "Low Revenue",
        "Per_Capita_Exp_TOTAL_PY": 11182,
        "ADM": 237,
        "P_EDV_Vis": 452,
        "P_EDV_Vis_HOSP": 183,
        "P_CT_VIS": 676,
        "P_MRI_VIS": 338,
        "P_EM_Total": 15065,
        "P_EM_PCP_Vis": 6291,
        "P_EM_SP_Vis": 7776,
        "P_SNF_ADM": 35,
        "Sav_rate": 0.0757
    },
    {
        "ACO_ID": "A1001",
        "ACO_Name": "Palm Beach Accountable Care Organization",
        "ACO_State": "FL",
        "performance_year": 2021,
        "Agree_Type": "Renewal",
        "Agreement_Period_Num": 3,
        "Risk_Model": "Two-Sided",
        "N_AB": 89403,
        "MinSavPerc": 0.02,
        "BnchmkMinExp": 84231357,
        "GenSaveLoss": 84231357,
        "EarnSaveLoss": 63173518,
        "QualScore": 100.0,
        "ABtotBnchmk": 1084231357,
        "ABtotExp": 1000000000,
        "FinalShareRate": 0.75,
        "RevLossLimit": 0.0,
        "Rev_Exp_Cat": "Low Revenue",
        "Per_Capita_Exp_TOTAL_PY": 11340,
        "ADM": 237,
        "P_EDV_Vis": 499,
        "P_EDV_Vis_HOSP": 187,
        "P_CT_VIS": 815,
        "P_MRI_VIS": 405,
        "P_EM_Total": 15707,
        "P_EM_PCP_Vis": 5960,
        "P_EM_SP_Vis": 8547,
        "P_SNF_ADM": 34,
        "Sav_rate": 0.0717
    },
    {
        "ACO_ID": "A1006",
        "ACO_Name": "Hackensack Alliance ACO",
        "ACO_State": "NJ, NY",
        "performance_year": 2020,
        "Agree_Type": "Renewal",
        "Agreement_Period_Num": 3,
        "Risk_Model": "One-Sided",
        "N_AB": 33798,
        "MinSavPerc": 0.0236,
        "BnchmkMinExp": -1917725,
        "GenSaveLoss": 0,
        "EarnSaveLoss": 0,
        "QualScore": 97.81,
        "ABtotBnchmk": 383064284,
        "ABtotExp": 384982008,
        "FinalShareRate": 0.39,
        "RevLossLimit": 0.0,
        "Rev_Exp_Cat": "Low Revenue",
        "Per_Capita_Exp_TOTAL_PY": 11784,
        "ADM": 248,
        "P_EDV_Vis": 454,
        "P_EDV_Vis_HOSP": 197,
        "P_CT_VIS": 641,
        "P_MRI_VIS": 250,
        "P_EM_Total": 11500,
        "P_EM_PCP_Vis": 4011,
        "P_EM_SP_Vis": 6683,
        "P_SNF_ADM": 66,
        "Sav_rate": -0.005
    }
]

ACO_ANALYTICS = [
    {
        "aco_id": "A1001",
        "aco_name": "Palm Beach Accountable Care Organization",
        "aco_state": "FL",
        "performance_year": 2020,
        "quality_score": 100.0,
        "previous_quality_score": None,
        "quality_change_yoy": None,
        "quality_change_yoy_pct": None,
        "quality_gap_to_100": 0.0,
        "quality_performance_category": "Excellent",
        "quality_status": "1",
        "cahps_1": None,
        "cahps_2": None,
        "cahps_3": None,
        "cahps_4": None,
        "cahps_5": None,
        "cahps_6": None,
        "cahps_7": None,
        "cahps_8": None,
        "cahps_9": None,
        "cahps_11": None,
        "assigned_beneficiaries": 80864,
        "previous_assigned_beneficiaries": None,
        "beneficiary_change_yoy": None,
        "beneficiary_change_yoy_pct": None,
        "age_0_64_pct": 6.17,
        "age_65_74_pct": 42.64,
        "age_75_84_pct": 35.41,
        "age_85_plus_pct": 15.79,
        "female_pct": 57.90,
        "male_pct": 42.10,
        "disabled_pct": 4.98,
        "esrd_pct": 0.64,
        "dual_eligible_pct": None,
        "average_available_risk_score": 1.07125,
        "risk_profile_category": "Very High"
    },
    {
        "aco_id": "A1001",
        "aco_name": "Palm Beach Accountable Care Organization",
        "aco_state": "FL",
        "performance_year": 2021,
        "quality_score": 100.0,
        "previous_quality_score": 100.0,
        "quality_change_yoy": 0.0,
        "quality_change_yoy_pct": 0.0,
        "quality_gap_to_100": 0.0,
        "quality_performance_category": "Excellent",
        "quality_status": "1",
        "cahps_1": 83.33,
        "cahps_2": 93.19,
        "cahps_3": 92.53,
        "cahps_4": 76.0,
        "cahps_5": 65.15,
        "cahps_6": 63.58,
        "cahps_7": 74.89,
        "cahps_8": 87.97,
        "cahps_9": 90.29,
        "cahps_11": 19.24,
        "assigned_beneficiaries": 89403,
        "previous_assigned_beneficiaries": 80864,
        "beneficiary_change_yoy": 8539,
        "beneficiary_change_yoy_pct": 10.56,
        "age_0_64_pct": 5.99,
        "age_65_74_pct": 43.24,
        "age_75_84_pct": 35.57,
        "age_85_plus_pct": 15.20,
        "female_pct": 57.51,
        "male_pct": 42.49,
        "disabled_pct": 5.18,
        "esrd_pct": 0.56,
        "dual_eligible_pct": 6.48,
        "average_available_risk_score": 1.071,
        "risk_profile_category": "Very High"
    }
]

ACO_ALERT = [
    {
        "ACO_ID": "A1199",
        "performance_year": 2023,
        "alert_type": "Utilization Anomaly",
        "severity": "HIGH",
        "message": "quality_change_yoy flagged as anomalous (score 2.52)"
    },
    {
        "ACO_ID": "A1241",
        "performance_year": 2023,
        "alert_type": "Utilization Anomaly",
        "severity": "HIGH",
        "message": "expenditure_variance_pct flagged as anomalous (score 2.16)"
    }
]

ACO_ANOMALIES = [
    {
        "ACO_ID": "A1199",
        "performance_year": 2023,
        "anomaly_score": 2.52065356775127,
        "top_anomalous_metric": "quality_change_yoy",
        "severity": "HIGH"
    },
    {
        "ACO_ID": "A1241",
        "performance_year": 2023,
        "anomaly_score": 2.15904052843946,
        "top_anomalous_metric": "expenditure_variance_pct",
        "severity": "HIGH"
    }
]

ACO_ANOMALY_FEATURES = [
    {
        "ACO_ID": "A1001",
        "performance_year": 2023,
        "savings_yoy_change_pct": 60.08,
        "expenditure_variance_pct": -9.09,
        "quality_change_yoy": 0.5,
        "ed_utilization_change_yoy": -0.032,
        "admission_change_yoy": -0.023,
        "em_utilization_change_yoy": -0.038,
        "advanced_imaging_change_yoy": -0.110,
        "ACO_State": "FL",
        "readmission_proxy_rate_yoy_change": 0.0
    },
    {
        "ACO_ID": "A1026",
        "performance_year": 2023,
        "savings_yoy_change_pct": None,
        "expenditure_variance_pct": -6.01,
        "quality_change_yoy": -10.22,
        "ed_utilization_change_yoy": -0.050,
        "admission_change_yoy": -0.054,
        "em_utilization_change_yoy": -0.030,
        "advanced_imaging_change_yoy": -0.134,
        "ACO_State": None,
        "readmission_proxy_rate_yoy_change": 0.0
    }
]

ACO_RISK_SCORES = [
    {
        "ACO_ID": "A1001",
        "performance_year": 2023,
        "ml_anomaly_score": 1.05284,
        "risk_level": "LOW",
        "is_anomaly": False
    },
    {
        "ACO_ID": "A1026",
        "performance_year": 2023,
        "ml_anomaly_score": 1.36557,
        "risk_level": "MEDIUM",
        "is_anomaly": False
    },
    {
        "ACO_ID": "A1199",
        "performance_year": 2023,
        "ml_anomaly_score": 2.52065,
        "risk_level": "HIGH",
        "is_anomaly": True
    }
]

ACO_SEGMENTATION_RESULTS = [
    {
        "id": 1,
        "ACO_ID": "A1001",
        "performance_year": 2020,
        "cluster_id": 0,
        "performance_segment": "High Performing",
        "performance_index": 0.8537,
        "SavingsLossPct": 7.569,
        "ExpenditureVariancePct": -7.569,
        "utilization_score": 3.234,
        "quality_score": 100.0
    },
    {
        "id": 2,
        "ACO_ID": "A1001",
        "performance_year": 2021,
        "cluster_id": 0,
        "performance_segment": "High Performing",
        "performance_index": 0.8537,
        "SavingsLossPct": 7.166,
        "ExpenditureVariancePct": -7.166,
        "utilization_score": 3.026,
        "quality_score": 100.0
    },
    {
        "id": 3,
        "ACO_ID": "A3458",
        "performance_year": 2021,
        "cluster_id": 1,
        "performance_segment": "High Performing",
        "performance_index": 0.912,
        "SavingsLossPct": 12.979,
        "ExpenditureVariancePct": -12.979,
        "utilization_score": 2.85,
        "quality_score": 96.87
    }
]

ACO_SEGMENTATION_ML_FEATURES = [
    {
        "id": 1,
        "ACO_ID": "A1001",
        "performance_year": 2020,
        "SavingsLossPct": 7.5693,
        "ExpenditureVariancePct": -7.5693,
        "PMPM": 911.44,
        "BenchmarkPMPM": 986.08,
        "FinancialGap": -72427998,
        "GenSaveLossYoYPct": None,
        "quality_score": 100.0,
        "quality_change_yoy": None,
        "quality_gap_to_100": 0.0,
        "utilization_score": 3.2346,
        "ed_visits_per_beneficiary": 0.00558,
        "admissions_per_beneficiary": 0.00293,
        "advanced_imaging_per_beneficiary": 0.01254,
        "assigned_beneficiaries": 80864,
        "disabled_pct": 4.98,
        "esrd_pct": 0.64,
        "average_available_risk_score": 1.07125
    },
    {
        "id": 2,
        "ACO_ID": "A1001",
        "performance_year": 2021,
        "SavingsLossPct": 7.1665,
        "ExpenditureVariancePct": -7.1665,
        "PMPM": 1017.04,
        "BenchmarkPMPM": 1095.55,
        "FinancialGap": -84231357,
        "GenSaveLossYoYPct": 16.30,
        "quality_score": 100.0,
        "quality_change_yoy": 0.0,
        "quality_gap_to_100": 0.0,
        "utilization_score": 3.0262,
        "ed_visits_per_beneficiary": 0.00558,
        "admissions_per_beneficiary": 0.00265,
        "advanced_imaging_per_beneficiary": 0.01365,
        "assigned_beneficiaries": 89403,
        "disabled_pct": 5.18,
        "esrd_pct": 0.56,
        "average_available_risk_score": 1.071
    }
]

ACO_UTILIZATION_ML_FEATURES = [
    {
        "ACO_ID": "A1001",
        "performance_year": 2020,
        "N_AB": 80864,
        "P_EDV_Vis": 452.0,
        "P_EDV_Vis_HOSP": 183.0,
        "ADM": 237.0,
        "ADM_S_Trm": 220.0,
        "P_SNF_ADM": 35.0,
        "P_CT_VIS": 676.0,
        "P_MRI_VIS": 338.0,
        "P_EM_Total": 15065.0,
        "P_EM_PCP_Vis": 6291.0,
        "P_EM_SP_Vis": 7776.0,
        "ed_visits_per_beneficiary": 0.005589,
        "admissions_per_beneficiary": 0.002931,
        "emergency_utilization_rate": 5.5896,
        "admission_rate_per_1000": 2.9308,
        "utilization_score": 3.23468,
        "utilization_category": "Low",
        "high_utilization_flag": False,
        "low_utilization_flag": True
    },
    {
        "ACO_ID": "A1001",
        "performance_year": 2021,
        "N_AB": 89403,
        "P_EDV_Vis": 499.0,
        "P_EDV_Vis_HOSP": 187.0,
        "ADM": 237.0,
        "ADM_S_Trm": 221.0,
        "P_SNF_ADM": 34.0,
        "P_CT_VIS": 815.0,
        "P_MRI_VIS": 405.0,
        "P_EM_Total": 15707.0,
        "P_EM_PCP_Vis": 5960.0,
        "P_EM_SP_Vis": 8547.0,
        "ed_visits_per_beneficiary": 0.005581,
        "admissions_per_beneficiary": 0.002651,
        "emergency_utilization_rate": 5.5814,
        "admission_rate_per_1000": 2.6509,
        "ed_utilization_change_yoy": -0.00146,
        "admission_change_yoy": -0.0955,
        "em_utilization_change_yoy": -0.0569,
        "advanced_imaging_change_yoy": 0.0882,
        "utilization_score": 3.02626,
        "utilization_category": "Low",
        "high_utilization_flag": False,
        "low_utilization_flag": True
    }
]

ACO_FINANCIAL_ML_TRAINING = [
    {
        "ACO_ID": "A3458",
        "feature_year": 2020,
        "target_year": 2021,
        "ABtotBnchmk": 70689610,
        "ABtotExp": 61514494,
        "GenSaveLoss": 9175117,
        "UpdatedBnchmk": 10587,
        "HistBnchmk": 11795,
        "EarnSaveLoss": 6665770,
        "FinalShareRate": 0.73,
        "N_AB": 6819,
        "SavingsLossPct": 12.9794,
        "ExpenditureVariancePct": -12.9794,
        "PMPM": 751.75,
        "BenchmarkPMPM": 863.88,
        "FinancialGap": -9175116.0,
        "QualScore": 96.87,
        "Risk_Model": "Two-Sided",
        "Rev_Exp_Cat": "Low Revenue"
    },
    {
        "ACO_ID": "A5275",
        "feature_year": 2023,
        "target_year": 2024,
        "ABtotBnchmk": 78628487,
        "ABtotExp": 78923302,
        "GenSaveLoss": 0,
        "UpdatedBnchmk": 9728,
        "HistBnchmk": 9167,
        "EarnSaveLoss": 0,
        "FinalShareRate": 40.0,
        "N_AB": 8218,
        "SavingsLossPct": 0.0,
        "ExpenditureVariancePct": 0.3749,
        "PMPM": 800.31,
        "BenchmarkPMPM": 797.32,
        "FinancialGap": 294815.0,
        "QualScore": 74.26,
        "Risk_Model": "One-Sided",
        "Rev_Exp_Cat": "High Revenue"
    }
]

BHAVYA_PROVIDER_ACO_FEATURES_FINAL = [
    {
        "id": 17401,
        "rndrng_npi": 1013478668,
        "rndrng_prvdr_last_org_name": "Thota",
        "rndrng_prvdr_first_name": "Pavankumar",
        "rndrng_prvdr_city": "New Orleans",
        "rndrng_prvdr_state_abrvtn": "LA",
        "rndrng_prvdr_type": "Internal Medicine",
        "tot_benes": 21,
        "tot_srvcs": 22.0,
        "tot_sbmtd_chrg": 28451.0,
        "tot_mdcr_alowd_amt": 3639.8,
        "tot_mdcr_pymt_amt": 2900.02,
        "tot_mdcr_stdzd_amt": 2927.0,
        "aco_id": "ACO_005",
        "year": 2024,
        "services_per_beneficiary": 1.0476,
        "payment_per_beneficiary": 138.10,
        "utilization_score": 0.02012,
        "cost_score": 0.52159,
        "provider_segment": "MODERATE_BALANCED",
        "high_utilization_flag": False,
        "high_cost_flag": False
    },
    {
        "id": 17402,
        "rndrng_npi": 1013480391,
        "rndrng_prvdr_last_org_name": "Kurlenda",
        "rndrng_prvdr_first_name": "Maria",
        "rndrng_prvdr_city": "Grand Rapids",
        "rndrng_prvdr_state_abrvtn": "MI",
        "rndrng_prvdr_type": "Nurse Practitioner",
        "tot_benes": 44,
        "tot_srvcs": 107.0,
        "tot_sbmtd_chrg": 9447.92,
        "tot_mdcr_alowd_amt": 4460.92,
        "tot_mdcr_pymt_amt": 2741.45,
        "tot_mdcr_stdzd_amt": 3744.49,
        "aco_id": "ACO_004",
        "year": 2020,
        "services_per_beneficiary": 2.4318,
        "payment_per_beneficiary": 62.31,
        "utilization_score": 0.4869,
        "cost_score": 0.1132,
        "provider_segment": "MODERATE_BALANCED",
        "high_utilization_flag": False,
        "high_cost_flag": False
    }
]

SERVICE_METRICS = [
    {
        "ACO_ID": "A1001",
        "Year": 2021,
        "HCPCS_Cd": "99214",
        "HCPCS_Desc": "Office or other outpatient visit for the evaluation and management of an established patient",
        "Place_Of_Srvc": "Office",
        "service_category": "Evaluation & Management",
        "provider_count": 142,
        "beneficiary_count": 15420,
        "service_volume": 28400,
        "total_submitted_charge": 4260000.0,
        "avg_allowed_amount": 132.50,
        "total_payment": 3010400.0,
        "avg_payment": 106.00,
        "avg_payment_per_beneficiary": 195.23,
        "high_cost_service": False,
        "high_utilization_service": True,
        "service_performance_segment": "HIGH_VOLUME_STANDARD_COST"
    }
]
