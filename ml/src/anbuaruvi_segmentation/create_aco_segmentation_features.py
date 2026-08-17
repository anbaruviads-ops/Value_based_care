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
    print("Check your .env file.")
    sys.exit(1)

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# ============================================================
# TABLES
# ============================================================

FINANCIAL_TABLE = "aco_financial_ml_training"

UTILIZATION_TABLE = "aco_utilization_ml_features"

QUALITY_TABLE = "aco_analytics"

TARGET_TABLE = "aco_segmentation_ml_features"

PAGE_SIZE = 1000


# ============================================================
# FINANCIAL COLUMNS
# ============================================================

FINANCIAL_COLUMNS = [
    "ACO_ID",
    "feature_year",

    "SavingsLossPct",
    "ExpenditureVariancePct",
    "PMPM",
    "BenchmarkPMPM",
    "FinancialGap",
    "GenSaveLossYoYPct",
]


# ============================================================
# UTILIZATION COLUMNS
# ============================================================

UTILIZATION_COLUMNS = [
    "ACO_ID",
    "performance_year",

    "utilization_score",

    "ed_visits_per_beneficiary",
    "admissions_per_beneficiary",
    "advanced_imaging_per_beneficiary",
    "em_visit_intensity",
    "snf_admissions_per_beneficiary",

    "ed_utilization_change_yoy",
    "admission_change_yoy",
    "em_utilization_change_yoy",
    "advanced_imaging_change_yoy",
]


# ============================================================
# QUALITY / BENEFICIARY / RISK COLUMNS
# ============================================================

QUALITY_COLUMNS = [
    "aco_id",
    "performance_year",

    "quality_score",
    "quality_change_yoy",
    "quality_change_yoy_pct",
    "quality_gap_to_100",

    "assigned_beneficiaries",
    "beneficiary_change_yoy_pct",

    "disabled_pct",
    "esrd_pct",
    "dual_eligible_pct",

    "average_available_risk_score",
    "risk_score_change_pct",
]


# ============================================================
# FETCH ALL ROWS
# ============================================================

def fetch_all_rows(table_name, columns):

    print()
    print("=" * 70)
    print(f"FETCHING: {table_name}")
    print("=" * 70)

    all_rows = []

    offset = 0

    while True:

        try:

            response = (
                supabase
                .table(table_name)
                .select(",".join(columns))
                .range(
                    offset,
                    offset + PAGE_SIZE - 1
                )
                .execute()
            )

        except Exception as e:

            print()
            print("ERROR FETCHING DATA")
            print(e)
            sys.exit(1)

        rows = response.data

        if not rows:
            break

        all_rows.extend(rows)

        print(
            f"{table_name}: retrieved "
            f"{len(all_rows)} rows"
        )

        if len(rows) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

        time.sleep(0.1)

    print(
        f"Total rows retrieved: "
        f"{len(all_rows)}"
    )

    return pd.DataFrame(all_rows)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

def convert_numeric(df, columns):

    for col in columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


# ============================================================
# CLEAN ACO IDs
# ============================================================

def clean_aco_id(df, column):

    if column not in df.columns:
        return df

    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
    )

    return df


# ============================================================
# CREATE SEGMENTATION FEATURES
# ============================================================

def create_segmentation_features(
    financial_df,
    utilization_df,
    quality_df
):

    print()
    print("=" * 70)
    print("STEP 7E — CREATING SEGMENTATION FEATURES")
    print("=" * 70)

    # ========================================================
    # CLEAN ACO IDS
    # ========================================================

    financial_df = clean_aco_id(
        financial_df,
        "ACO_ID"
    )

    utilization_df = clean_aco_id(
        utilization_df,
        "ACO_ID"
    )

    quality_df = clean_aco_id(
        quality_df,
        "aco_id"
    )

    # ========================================================
    # CONVERT YEARS
    # ========================================================

    financial_df["feature_year"] = pd.to_numeric(
        financial_df["feature_year"],
        errors="coerce"
    )

    utilization_df["performance_year"] = pd.to_numeric(
        utilization_df["performance_year"],
        errors="coerce"
    )

    quality_df["performance_year"] = pd.to_numeric(
        quality_df["performance_year"],
        errors="coerce"
    )

    # ========================================================
    # CONVERT NUMERIC FEATURES
    # ========================================================

    financial_numeric = [
        col
        for col in FINANCIAL_COLUMNS
        if col not in ["ACO_ID", "feature_year"]
    ]

    utilization_numeric = [
        col
        for col in UTILIZATION_COLUMNS
        if col not in ["ACO_ID", "performance_year"]
    ]

    quality_numeric = [
        col
        for col in QUALITY_COLUMNS
        if col not in ["aco_id", "performance_year"]
    ]

    financial_df = convert_numeric(
        financial_df,
        financial_numeric
    )

    utilization_df = convert_numeric(
        utilization_df,
        utilization_numeric
    )

    quality_df = convert_numeric(
        quality_df,
        quality_numeric
    )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    financial_duplicates = financial_df.duplicated(
        subset=[
            "ACO_ID",
            "feature_year"
        ]
    ).sum()

    utilization_duplicates = utilization_df.duplicated(
        subset=[
            "ACO_ID",
            "performance_year"
        ]
    ).sum()

    quality_duplicates = quality_df.duplicated(
        subset=[
            "aco_id",
            "performance_year"
        ]
    ).sum()

    print()
    print("Duplicate checks:")
    print(
        f"Financial:    {financial_duplicates}"
    )
    print(
        f"Utilization:  {utilization_duplicates}"
    )
    print(
        f"Quality:      {quality_duplicates}"
    )

    if financial_duplicates > 0:
        financial_df = financial_df.drop_duplicates(
            subset=[
                "ACO_ID",
                "feature_year"
            ],
            keep="first"
        )

    if utilization_duplicates > 0:
        utilization_df = utilization_df.drop_duplicates(
            subset=[
                "ACO_ID",
                "performance_year"
            ],
            keep="first"
        )

    if quality_duplicates > 0:
        quality_df = quality_df.drop_duplicates(
            subset=[
                "aco_id",
                "performance_year"
            ],
            keep="first"
        )

    # ========================================================
    # RENAME QUALITY ACO_ID
    # ========================================================

    quality_df = quality_df.rename(
        columns={
            "aco_id": "ACO_ID"
        }
    )

    # ========================================================
    # FINANCIAL YEAR -> PERFORMANCE YEAR
    #
    # IMPORTANT:
    # feature_year is used for segmentation.
    # target_year is NOT used.
    # ========================================================

    financial_df = financial_df.rename(
        columns={
            "feature_year": "performance_year"
        }
    )

    # ========================================================
    # MERGE FINANCIAL + UTILIZATION
    # ========================================================

    print()
    print("=" * 70)
    print("JOINING FINANCIAL + UTILIZATION")
    print("=" * 70)

    financial_utilization = pd.merge(
        financial_df,
        utilization_df,
        on=[
            "ACO_ID",
            "performance_year"
        ],
        how="inner",
        validate="one_to_one"
    )

    print(
        "Financial rows:",
        len(financial_df)
    )

    print(
        "Utilization rows:",
        len(utilization_df)
    )

    print(
        "Financial ∩ Utilization:",
        len(financial_utilization)
    )

    # ========================================================
    # MERGE QUALITY
    # ========================================================

    print()
    print("=" * 70)
    print("JOINING QUALITY")
    print("=" * 70)

    final_df = pd.merge(
        financial_utilization,
        quality_df,
        on=[
            "ACO_ID",
            "performance_year"
        ],
        how="inner",
        validate="one_to_one"
    )

    print(
        "After quality join:",
        len(final_df)
    )

    # ========================================================
    # SORT
    # ========================================================

    final_df = final_df.sort_values(
        [
            "ACO_ID",
            "performance_year"
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # FINAL COLUMN ORDER
    # ========================================================

    final_columns = [

        # ------------------------------------
        # Keys
        # ------------------------------------

        "ACO_ID",
        "performance_year",

        # ------------------------------------
        # Financial
        # ------------------------------------

        "SavingsLossPct",
        "ExpenditureVariancePct",
        "PMPM",
        "BenchmarkPMPM",
        "FinancialGap",
        "GenSaveLossYoYPct",

        # ------------------------------------
        # Quality
        # ------------------------------------

        "quality_score",
        "quality_change_yoy",
        "quality_change_yoy_pct",
        "quality_gap_to_100",

        # ------------------------------------
        # Utilization
        # ------------------------------------

        "utilization_score",

        "ed_visits_per_beneficiary",
        "admissions_per_beneficiary",
        "advanced_imaging_per_beneficiary",
        "em_visit_intensity",
        "snf_admissions_per_beneficiary",

        "ed_utilization_change_yoy",
        "admission_change_yoy",
        "em_utilization_change_yoy",
        "advanced_imaging_change_yoy",

        # ------------------------------------
        # Beneficiary / Risk
        # ------------------------------------

        "assigned_beneficiaries",
        "beneficiary_change_yoy_pct",

        "disabled_pct",
        "esrd_pct",
        "dual_eligible_pct",

        "average_available_risk_score",
        "risk_score_change_pct",
    ]

    # ========================================================
    # CHECK REQUIRED COLUMNS
    # ========================================================

    missing_columns = [
        col
        for col in final_columns
        if col not in final_df.columns
    ]

    if missing_columns:

        print()
        print("ERROR: Missing required columns:")

        for col in missing_columns:
            print(
                f"  - {col}"
            )

        sys.exit(1)

    final_df = final_df[
        final_columns
    ].copy()

    return final_df


# ============================================================
# VALIDATE FINAL FEATURES
# ============================================================

def validate_features(df):

    print()
    print("=" * 70)
    print("STEP 7E — FINAL VALIDATION")
    print("=" * 70)

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        f"Unique ACOs: "
        f"{df['ACO_ID'].nunique()}"
    )

    print(
        f"Years: "
        f"{sorted(df['performance_year'].dropna().unique())}"
    )

    duplicates = df.duplicated(
        subset=[
            "ACO_ID",
            "performance_year"
        ]
    ).sum()

    print(
        f"Duplicate ACO-Year rows: "
        f"{duplicates}"
    )

    print()
    print("Rows by year:")

    print(
        df["performance_year"]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # MISSING VALUES
    # ========================================================

    print()
    print("Missing-value percentage:")

    feature_columns = [
        col
        for col in df.columns
        if col not in [
            "ACO_ID",
            "performance_year"
        ]
    ]

    for col in feature_columns:

        missing_pct = (
            df[col].isna().mean()
            * 100
        )

        if missing_pct > 0:

            print(
                f"{col:45s} "
                f"{missing_pct:.2f}% missing"
            )

    # ========================================================
    # INVALID NUMBERS
    # ========================================================

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    invalid_count = 0

    for col in numeric_columns:

        invalid = (
            ~np.isfinite(
                df[col].dropna()
            )
        ).sum()

        invalid_count += invalid

    print()
    print(
        f"Invalid numeric values: "
        f"{invalid_count}"
    )

    # ========================================================
    # UTILIZATION SCORE
    # ========================================================

    print()
    print("Utilization score statistics:")

    print(
        df["utilization_score"].describe()
    )

    # ========================================================
    # QUALITY SCORE
    # ========================================================

    print()
    print("Quality score statistics:")

    print(
        df["quality_score"].describe()
    )

    # ========================================================
    # FINANCIAL
    # ========================================================

    print()
    print("Savings/Loss statistics:")

    print(
        df["SavingsLossPct"].describe()
    )

    # ========================================================
    # FINAL VALIDATION CONDITIONS
    # ========================================================

    if duplicates != 0:

        print()
        print(
            "ERROR: Duplicate ACO-Year rows found."
        )

        sys.exit(1)

    if len(df) != 1713:

        print()
        print(
            "WARNING: Expected approximately "
            "1713 rows based on the current "
            "ACO-Year overlap."
        )

        print(
            f"Actual rows: {len(df)}"
        )

    print()
    print("FINAL VALIDATION PASSED.")


# ============================================================
# CLEAN RECORDS FOR JSON
# ============================================================

def clean_records(df):

    clean_df = df.copy()

    # Replace infinity
    clean_df = clean_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Convert year explicitly to integer
    clean_df["performance_year"] = (
        pd.to_numeric(
            clean_df["performance_year"],
            errors="coerce"
        )
        .round()
        .astype("Int64")
    )

    # Check invalid years
    if clean_df["performance_year"].isna().any():

        print(
            "ERROR: performance_year contains "
            "invalid values."
        )

        sys.exit(1)

    # Convert to Python records
    records = clean_df.to_dict(
        orient="records"
    )

    # ========================================================
    # CONVERT NUMPY VALUES TO PYTHON VALUES
    # ========================================================

    for record in records:

        for key, value in record.items():

            if value is None:
                continue

            # NumPy integer
            if isinstance(
                value,
                np.integer
            ):

                record[key] = int(value)

            # NumPy float
            elif isinstance(
                value,
                np.floating
            ):

                if np.isfinite(value):

                    record[key] = float(value)

                else:

                    record[key] = None

            # Python float
            elif isinstance(
                value,
                float
            ):

                if np.isfinite(value):

                    record[key] = float(value)

                else:

                    record[key] = None

            # pandas missing value
            elif pd.isna(value):

                record[key] = None

    return records


# ============================================================
# INSERT / UPSERT
# ============================================================

def insert_data(df):

    print()
    print("=" * 70)
    print("INSERTING INTO SUPABASE")
    print("=" * 70)

    records = clean_records(df)

    print(
        f"Clean records ready: "
        f"{len(records)}"
    )

    batch_size = 500

    total = len(records)

    for start in range(
        0,
        total,
        batch_size
    ):

        batch = records[
            start:start + batch_size
        ]

        try:

            (
                supabase
                .table(TARGET_TABLE)
                .upsert(
                    batch,
                    on_conflict=(
                        "ACO_ID,"
                        "performance_year"
                    )
                )
                .execute()
            )

        except Exception as e:

            print()
            print(
                "ERROR INSERTING DATA"
            )

            print(e)

            sys.exit(1)

        end = min(
            start + batch_size,
            total
        )

        print(
            f"Inserted/upserted "
            f"{end} / {total}"
        )

    print()
    print(
        "INSERT/UPSERT COMPLETE"
    )


# ============================================================
# VERIFY SUPABASE TABLE
# ============================================================

def verify_supabase():

    print()
    print("=" * 70)
    print("VERIFYING SUPABASE TARGET TABLE")
    print("=" * 70)

    try:

        response = (
            supabase
            .table(TARGET_TABLE)
            .select(
                "ACO_ID,performance_year"
            )
            .range(
                0,
                PAGE_SIZE - 1
            )
            .execute()
        )

        rows = response.data

        print(
            f"Successfully queried "
            f"{TARGET_TABLE}"
        )

        print(
            f"First page rows: "
            f"{len(rows)}"
        )

    except Exception as e:

        print(
            "ERROR VERIFYING TARGET TABLE"
        )

        print(e)

        sys.exit(1)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("STEP 7E — ACO SEGMENTATION FEATURE ENGINEERING")
    print("=" * 70)

    # ========================================================
    # FETCH FINANCIAL
    # ========================================================

    financial_df = fetch_all_rows(
        FINANCIAL_TABLE,
        FINANCIAL_COLUMNS
    )

    if financial_df.empty:

        print(
            "ERROR: Financial table is empty."
        )

        sys.exit(1)

    # ========================================================
    # FETCH UTILIZATION
    # ========================================================

    utilization_df = fetch_all_rows(
        UTILIZATION_TABLE,
        UTILIZATION_COLUMNS
    )

    if utilization_df.empty:

        print(
            "ERROR: Utilization table is empty."
        )

        sys.exit(1)

    # ========================================================
    # FETCH QUALITY
    # ========================================================

    quality_df = fetch_all_rows(
        QUALITY_TABLE,
        QUALITY_COLUMNS
    )

    if quality_df.empty:

        print(
            "ERROR: Quality table is empty."
        )

        sys.exit(1)

    # ========================================================
    # CREATE FEATURES
    # ========================================================

    feature_df = create_segmentation_features(
        financial_df,
        utilization_df,
        quality_df
    )

    # ========================================================
    # PRINT COLUMNS
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL SEGMENTATION FEATURE COLUMNS")
    print("=" * 70)

    for i, col in enumerate(
        feature_df.columns,
        start=1
    ):

        print(
            f"{i:02d}. {col}"
        )

    # ========================================================
    # VALIDATE
    # ========================================================

    validate_features(
        feature_df
    )

    # ========================================================
    # READY
    # ========================================================

    print()
    print("=" * 70)
    print("READY TO WRITE TO SUPABASE")
    print("=" * 70)

    print(
        f"Target table: "
        f"{TARGET_TABLE}"
    )

    print(
        f"Rows: "
        f"{len(feature_df)}"
    )

    print(
        f"Columns: "
        f"{len(feature_df.columns)}"
    )

    # ========================================================
    # INSERT
    # ========================================================

    insert_data(
        feature_df
    )

    # ========================================================
    # VERIFY
    # ========================================================

    verify_supabase()

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 7E COMPLETE")
    print("=" * 70)

    print(
        f"Created/updated: "
        f"{TARGET_TABLE}"
    )

    print(
        "Source tables were not modified."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()