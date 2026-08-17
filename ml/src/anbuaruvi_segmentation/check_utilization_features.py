import os
from dotenv import load_dotenv
from supabase import create_client


# ============================================================
# 1. LOAD SUPABASE CONNECTION
# ============================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        "SUPABASE_URL or SUPABASE_KEY is missing from .env"
    )

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# ============================================================
# 2. TABLES TO INSPECT
# ============================================================

TABLES = {
    "fact_aco_performance": [
        "P_EDV_Vis",
        "P_EDV_Vis_HOSP",
        "ADM",
        "ADM_S_Trm",
        "ADM_L_Trm",
        "ADM_Rehab",
        "ADM_Psych",
        "P_SNF_ADM",
        "SNF_LOS",
        "SNF_PayperStay",
        "chf_adm",
        "copd_adm",
        "prov_Rate_1000",
        "P_CT_VIS",
        "P_MRI_VIS",
        "P_EM_Total",
        "P_EM_PCP_Vis",
        "P_EM_SP_Vis",
        "P_Nurse_Vis",
        "P_FQHC_RHC_Vis"
    ],

    "aco_financial_ml_training": [
        "ADM",
        "ADM_S_Trm",
        "ADM_Rehab",
        "P_EDV_Vis",
        "P_EDV_Vis_HOSP",
        "P_CT_VIS",
        "P_MRI_VIS",
        "P_EM_Total",
        "P_EM_PCP_Vis",
        "P_EM_SP_Vis",
        "P_Nurse_Vis",
        "P_FQHC_RHC_Vis",
        "P_SNF_ADM",
        "SNF_LOS",
        "SNF_PayperStay",
        "chf_adm",
        "prov_Rate_1000"
    ],

    "aco_analytics": [
        # Possible utilization-related features
        "utilization_score",
        "high_utilization_flag",
        "low_utilization_flag",

        # These are not currently in the schema you showed,
        # but we search for them if they exist.
        "er_visits",
        "er_visits_per_beneficiary",
        "ed_visits",
        "ed_visits_per_beneficiary",
        "admissions",
        "admissions_per_beneficiary",
        "admission_rate",
        "readmission_rate",
        "utilization_change_yoy",
        "er_utilization_change",
        "admission_change",
        "readmission_change"
    ],

    "bhavya_provider_aco_features_final": [
        "tot_benes",
        "tot_srvcs",
        "med_tot_benes",
        "med_tot_srvcs",
        "services_per_beneficiary",
        "medical_services_per_beneficiary",
        "service_intensity_per_beneficiary",
        "risk_adjusted_services",
        "condition_adjusted_services",
        "services_per_condition_burden",
        "services_per_risk_score",
        "yoy_service_change_pct",
        "yoy_services_per_beneficiary_change_pct",
        "long_term_service_change_pct",
        "utilization_score",
        "high_utilization_flag",
        "low_utilization_flag"
    ]
}


# ============================================================
# 3. GET ACTUAL COLUMNS FROM EACH TABLE
# ============================================================

def get_actual_columns(table_name):

    response = (
        supabase
        .table(table_name)
        .select("*")
        .limit(1)
        .execute()
    )

    if not response.data:
        print(
            f"\nWARNING: {table_name} returned no rows."
        )
        return set()

    return set(response.data[0].keys())


# ============================================================
# 4. CHECK FEATURE AVAILABILITY
# ============================================================

results = []


print("\n")
print("=" * 80)
print("STEP 7A — UTILIZATION FEATURE AVAILABILITY CHECK")
print("=" * 80)


for table_name, candidates in TABLES.items():

    print("\n" + "-" * 80)
    print(f"TABLE: {table_name}")
    print("-" * 80)

    try:
        actual_columns = get_actual_columns(table_name)

    except Exception as e:

        print(
            f"ERROR reading {table_name}: {e}"
        )

        continue

    for feature in candidates:

        exists = feature in actual_columns

        if exists:
            status = "AVAILABLE"
        else:
            status = "NOT AVAILABLE"

        results.append({
            "table": table_name,
            "feature": feature,
            "available": exists
        })

        print(
            f"{feature:<45} {status}"
        )


# ============================================================
# 5. SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("SUMMARY")
print("=" * 80)


for table_name in TABLES.keys():

    table_results = [
        r for r in results
        if r["table"] == table_name
    ]

    available = [
        r["feature"]
        for r in table_results
        if r["available"]
    ]

    unavailable = [
        r["feature"]
        for r in table_results
        if not r["available"]
    ]

    print("\n")
    print(f"TABLE: {table_name}")

    print(
        f"Available utilization features: "
        f"{len(available)}"
    )

    for feature in available:
        print(f"  ✓ {feature}")

    print(
        f"Not available: "
        f"{len(unavailable)}"
    )


# ============================================================
# 6. CROSS-TABLE FEATURE SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("CROSS-TABLE UTILIZATION FEATURE SUMMARY")
print("=" * 80)


all_candidate_features = sorted(
    {
        feature
        for candidates in TABLES.values()
        for feature in candidates
    }
)


for feature in all_candidate_features:

    available_tables = [
        table_name
        for table_name, candidates in TABLES.items()
        if feature in candidates
        and feature in get_actual_columns(table_name)
    ]

    print(
        f"\n{feature}"
    )

    if available_tables:

        for table_name in available_tables:
            print(
                f"  ✓ {table_name}"
            )

    else:

        print(
            "  ✗ Not available in any table"
        )


print("\n")
print("=" * 80)
print("STEP 7A COMPLETE")
print("=" * 80)

print(
    "\nNo data was modified."
)

print(
    "No features were calculated."
)

print(
    "No rows were inserted into Supabase."
)