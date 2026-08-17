import os
import sys
import time
import numpy as np
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("ERROR: SUPABASE_URL or SUPABASE_KEY is missing.")
    sys.exit(1)

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

TABLE = "aco_utilization_ml_features"
PAGE_SIZE = 1000


# ============================================================
# FETCH ALL ROWS
# ============================================================

def fetch_all_rows():

    print("=" * 70)
    print("STEP 7C — ACO UTILIZATION FEATURE TABLE VALIDATION")
    print("=" * 70)

    all_rows = []
    offset = 0

    while True:

        response = (
            supabase
            .table(TABLE)
            .select("*")
            .range(offset, offset + PAGE_SIZE - 1)
            .execute()
        )

        rows = response.data

        if not rows:
            break

        all_rows.extend(rows)

        print(
            f"Retrieved {len(all_rows)} rows"
        )

        if len(rows) < PAGE_SIZE:
            break

        offset += PAGE_SIZE
        time.sleep(0.1)

    print()
    print(f"TOTAL ROWS RETRIEVED: {len(all_rows)}")

    return pd.DataFrame(all_rows)


# ============================================================
# VALIDATE
# ============================================================

def validate(df):

    print()
    print("=" * 70)
    print("1. BASIC TABLE VALIDATION")
    print("=" * 70)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print(
        f"Unique ACOs: {df['ACO_ID'].nunique()}"
    )

    print(
        f"Years: {sorted(df['performance_year'].dropna().unique())}"
    )


    # ========================================================
    # ACO-YEAR DUPLICATES
    # ========================================================

    print()
    print("=" * 70)
    print("2. ACO-YEAR UNIQUENESS")
    print("=" * 70)

    duplicates = df.duplicated(
        subset=["ACO_ID", "performance_year"]
    ).sum()

    print(
        f"Duplicate ACO-Year rows: {duplicates}"
    )

    if duplicates == 0:
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")


    # ========================================================
    # YEAR DISTRIBUTION
    # ========================================================

    print()
    print("=" * 70)
    print("3. YEAR DISTRIBUTION")
    print("=" * 70)

    print(
        df["performance_year"]
        .value_counts()
        .sort_index()
    )


    # ========================================================
    # UTILIZATION SCORE
    # ========================================================

    print()
    print("=" * 70)
    print("4. UTILIZATION SCORE")
    print("=" * 70)

    print(
        df["utilization_score"].describe()
    )

    invalid_score = (
        ~np.isfinite(
            pd.to_numeric(
                df["utilization_score"],
                errors="coerce"
            )
        )
    ).sum()

    print(
        f"Invalid utilization scores: {invalid_score}"
    )


    # ========================================================
    # UTILIZATION CATEGORY
    # ========================================================

    print()
    print("=" * 70)
    print("5. UTILIZATION CATEGORY")
    print("=" * 70)

    print(
        df["utilization_category"]
        .value_counts(dropna=False)
    )


    # ========================================================
    # FLAGS
    # ========================================================

    print()
    print("=" * 70)
    print("6. UTILIZATION FLAGS")
    print("=" * 70)

    print(
        "High utilization:"
    )

    print(
        df["high_utilization_flag"]
        .value_counts(dropna=False)
    )

    print()

    print(
        "Low utilization:"
    )

    print(
        df["low_utilization_flag"]
        .value_counts(dropna=False)
    )


    # ========================================================
    # IMPORTANT FEATURES
    # ========================================================

    print()
    print("=" * 70)
    print("7. IMPORTANT FEATURE MISSING VALUES")
    print("=" * 70)

    important_features = [

        "N_AB",

        "ed_visits_per_beneficiary",

        "hospital_ed_visits_per_beneficiary",

        "admissions_per_beneficiary",

        "snf_admissions_per_beneficiary",

        "advanced_imaging_per_beneficiary",

        "emergency_utilization_rate",

        "admission_rate_per_1000",

        "em_visit_intensity",

        "utilization_score",

        "utilization_category",

    ]

    for column in important_features:

        if column not in df.columns:

            print(
                f"{column:45s} COLUMN NOT FOUND"
            )

            continue

        missing = df[column].isna().sum()

        percentage = (
            missing / len(df) * 100
        )

        print(
            f"{column:45s} "
            f"{missing:5d} missing "
            f"({percentage:.2f}%)"
        )


    # ========================================================
    # YOY FEATURES
    # ========================================================

    print()
    print("=" * 70)
    print("8. YOY FEATURE CHECK")
    print("=" * 70)

    yoy_features = [

        "previous_utilization_year",

        "previous_ed_visits_per_beneficiary",

        "previous_admissions_per_beneficiary",

        "previous_em_visit_intensity",

        "previous_advanced_imaging_per_beneficiary",

        "ed_utilization_change_yoy",

        "admission_change_yoy",

        "em_utilization_change_yoy",

        "advanced_imaging_change_yoy",

    ]

    for column in yoy_features:

        if column in df.columns:

            missing = df[column].isna().sum()

            print(
                f"{column:45s} "
                f"{missing} missing"
            )


    # ========================================================
    # INVALID NUMERIC VALUES
    # ========================================================

    print()
    print("=" * 70)
    print("9. NaN / INFINITY CHECK")
    print("=" * 70)

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    total_nan = 0
    total_inf = 0

    for column in numeric_columns:

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        nan_count = values.isna().sum()

        inf_count = np.isinf(
            values.dropna()
        ).sum()

        total_nan += nan_count
        total_inf += inf_count

    print(f"Total NaN values: {total_nan}")
    print(f"Total Infinity values: {total_inf}")

    if total_inf == 0:
        print("Infinity check: PASS")
    else:
        print("Infinity check: FAIL")


    # ========================================================
    # COLUMN CHECK
    # ========================================================

    print()
    print("=" * 70)
    print("10. REQUIRED COLUMN CHECK")
    print("=" * 70)

    required_columns = [

        "ACO_ID",
        "performance_year",
        "N_AB",

        "ed_visits_per_beneficiary",
        "hospital_ed_visits_per_beneficiary",
        "admissions_per_beneficiary",
        "snf_admissions_per_beneficiary",

        "advanced_imaging_per_beneficiary",

        "emergency_utilization_rate",
        "admission_rate_per_1000",

        "ed_intensity",
        "admission_intensity",
        "em_visit_intensity",

        "previous_utilization_year",

        "ed_utilization_change_yoy",
        "admission_change_yoy",

        "utilization_score",
        "utilization_category",

        "high_utilization_flag",
        "low_utilization_flag",

    ]

    missing_columns = []

    for column in required_columns:

        if column not in df.columns:

            missing_columns.append(column)

    if not missing_columns:

        print("All required columns exist.")
        print("STATUS: PASS")

    else:

        print("Missing columns:")

        for column in missing_columns:
            print(f"  - {column}")

        print("STATUS: FAIL")


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 7C FINAL STATUS")
    print("=" * 70)

    if (
        len(df) > 0
        and duplicates == 0
        and total_inf == 0
        and not missing_columns
    ):

        print("STATUS: PASS")
        print()
        print(
            "aco_utilization_ml_features is ready "
            "for the ML pipeline."
        )

    else:

        print("STATUS: REVIEW REQUIRED")


# ============================================================
# MAIN
# ============================================================

def main():

    df = fetch_all_rows()

    if df.empty:

        print()
        print("ERROR: Feature table contains no rows.")
        sys.exit(1)

    validate(df)


if __name__ == "__main__":
    main()