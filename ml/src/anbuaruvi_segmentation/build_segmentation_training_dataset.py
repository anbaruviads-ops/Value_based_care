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

UTILIZATION_TABLE = "aco_utilization_ml_features"
FINANCIAL_TABLE = "aco_financial_ml_training"
QUALITY_TABLE = "aco_analytics"

PAGE_SIZE = 1000


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_FILE = "data/segmentation_training_dataset.csv"


# ============================================================
# FEATURES REQUIRED FOR SEGMENTATION
# ============================================================

FINANCIAL_FEATURES = [
    "SavingsLossPct",
    "ExpenditureVariancePct",
    "PMPM",
    "BenchmarkPMPM",
    "FinancialGap",
    "GenSaveLossYoYPct",
]


QUALITY_FEATURES = [
    "quality_score",
    "quality_change_yoy_pct",
    "quality_gap_to_100",
    "attention_area_count",
]


UTILIZATION_FEATURES = [
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
            print(f"ERROR retrieving {table_name}")
            print(e)
            sys.exit(1)

        rows = response.data

        if not rows:
            break

        all_rows.extend(rows)

        print(
            f"{table_name}: "
            f"retrieved {len(all_rows)} rows"
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
# SAFE NUMERIC CONVERSION
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
# VALIDATE ACO-YEAR UNIQUENESS
# ============================================================

def validate_unique(df, aco_column, year_column, table_name):

    duplicates = df.duplicated(
        subset=[
            aco_column,
            year_column
        ]
    ).sum()

    print()
    print(
        f"{table_name} duplicate ACO-Year rows: "
        f"{duplicates}"
    )

    if duplicates > 0:

        print(
            f"ERROR: {table_name} contains "
            f"duplicate ACO-Year rows."
        )

        duplicate_rows = df[
            df.duplicated(
                subset=[
                    aco_column,
                    year_column
                ],
                keep=False
            )
        ]

        print(duplicate_rows.head(20))

        sys.exit(1)


# ============================================================
# BUILD TRAINING DATASET
# ============================================================

def build_training_dataset(
    utilization_df,
    financial_df,
    quality_df
):

    print()
    print("=" * 70)
    print("STEP 8A — BUILDING SEGMENTATION TRAINING DATASET")
    print("=" * 70)

    # ========================================================
    # PREPARE UTILIZATION
    # ========================================================

    utilization = utilization_df.copy()

    utilization["performance_year"] = pd.to_numeric(
        utilization["performance_year"],
        errors="coerce"
    )

    utilization["ACO_ID"] = (
        utilization["ACO_ID"]
        .astype(str)
        .str.strip()
    )

    validate_unique(
        utilization,
        "ACO_ID",
        "performance_year",
        UTILIZATION_TABLE
    )

    # ========================================================
    # PREPARE FINANCIAL
    # ========================================================

    financial = financial_df.copy()

    financial["feature_year"] = pd.to_numeric(
        financial["feature_year"],
        errors="coerce"
    )

    financial["target_year"] = pd.to_numeric(
        financial["target_year"],
        errors="coerce"
    )

    financial["ACO_ID"] = (
        financial["ACO_ID"]
        .astype(str)
        .str.strip()
    )

    validate_unique(
        financial,
        "ACO_ID",
        "target_year",
        FINANCIAL_TABLE
    )

    # ========================================================
    # PREPARE QUALITY
    # ========================================================

    quality = quality_df.copy()

    quality["performance_year"] = pd.to_numeric(
        quality["performance_year"],
        errors="coerce"
    )

    quality["aco_id"] = (
        quality["aco_id"]
        .astype(str)
        .str.strip()
    )

    # Rename quality ACO_ID to common name
    quality = quality.rename(
        columns={
            "aco_id": "ACO_ID"
        }
    )

    validate_unique(
        quality,
        "ACO_ID",
        "performance_year",
        QUALITY_TABLE
    )

    # ========================================================
    # REMOVE INVALID KEYS
    # ========================================================

    utilization = utilization.dropna(
        subset=[
            "ACO_ID",
            "performance_year"
        ]
    )

    financial = financial.dropna(
        subset=[
            "ACO_ID",
            "target_year"
        ]
    )

    quality = quality.dropna(
        subset=[
            "ACO_ID",
            "performance_year"
        ]
    )

    # ========================================================
    # STANDARDIZE YEAR DATATYPE
    # ========================================================

    utilization["performance_year"] = (
        utilization["performance_year"]
        .astype(int)
    )

    financial["target_year"] = (
        financial["target_year"]
        .astype(int)
    )

    quality["performance_year"] = (
        quality["performance_year"]
        .astype(int)
    )

    # ========================================================
    # RENAME FINANCIAL YEAR
    # ========================================================

    financial = financial.rename(
        columns={
            "target_year": "performance_year"
        }
    )

    # ========================================================
    # PRINT SOURCE COUNTS
    # ========================================================

    print()
    print("-" * 70)
    print("SOURCE DATASET COUNTS")
    print("-" * 70)

    print(
        f"Utilization ACO-Year rows: "
        f"{len(utilization)}"
    )

    print(
        f"Financial ACO-Year rows: "
        f"{len(financial)}"
    )

    print(
        f"Quality ACO-Year rows: "
        f"{len(quality)}"
    )

    # ========================================================
    # FIND UTILIZATION ∩ FINANCIAL
    # ========================================================

    utilization_keys = set(
        zip(
            utilization["ACO_ID"],
            utilization["performance_year"]
        )
    )

    financial_keys = set(
        zip(
            financial["ACO_ID"],
            financial["performance_year"]
        )
    )

    quality_keys = set(
        zip(
            quality["ACO_ID"],
            quality["performance_year"]
        )
    )

    uf_keys = (
        utilization_keys
        & financial_keys
    )

    all_three_keys = (
        utilization_keys
        & financial_keys
        & quality_keys
    )

    print()
    print("-" * 70)
    print("ACO-YEAR OVERLAP")
    print("-" * 70)

    print(
        f"Utilization ∩ Financial: "
        f"{len(uf_keys)}"
    )

    print(
        f"Utilization ∩ Financial ∩ Quality: "
        f"{len(all_three_keys)}"
    )

    # ========================================================
    # INNER JOIN
    # ========================================================

    print()
    print("-" * 70)
    print("JOINING DATASETS")
    print("-" * 70)

    # --------------------------------------------------------
    # Utilization + Financial
    # --------------------------------------------------------

    merged = utilization.merge(
        financial,
        on=[
            "ACO_ID",
            "performance_year"
        ],
        how="inner",
        suffixes=(
            "_utilization",
            "_financial"
        )
    )

    print(
        f"After Utilization + Financial join: "
        f"{len(merged)} rows"
    )

    # --------------------------------------------------------
    # + Quality
    # --------------------------------------------------------

    merged = merged.merge(
        quality,
        on=[
            "ACO_ID",
            "performance_year"
        ],
        how="inner",
        suffixes=(
            "",
            "_quality"
        )
    )

    print(
        f"After + Quality join: "
        f"{len(merged)} rows"
    )

    # ========================================================
    # VERIFY EXPECTED ROW COUNT
    # ========================================================

    expected_rows = len(all_three_keys)

    print()
    print(
        f"Expected common ACO-Year rows: "
        f"{expected_rows}"
    )

    print(
        f"Actual training rows: "
        f"{len(merged)}"
    )

    if len(merged) != expected_rows:

        print()
        print(
            "ERROR: Final row count does not "
            "match the calculated ACO-Year overlap."
        )

        sys.exit(1)

    # ========================================================
    # VERIFY NO DUPLICATES
    # ========================================================

    duplicate_count = merged.duplicated(
        subset=[
            "ACO_ID",
            "performance_year"
        ]
    ).sum()

    print(
        f"Duplicate ACO-Year rows: "
        f"{duplicate_count}"
    )

    if duplicate_count > 0:

        print(
            "ERROR: Duplicate ACO-Year rows "
            "exist after joining."
        )

        sys.exit(1)

    # ========================================================
    # SELECT FINAL FEATURES
    # ========================================================

    final_columns = [
        "ACO_ID",
        "performance_year",
    ]

    final_columns.extend(
        FINANCIAL_FEATURES
    )

    final_columns.extend(
        QUALITY_FEATURES
    )

    final_columns.extend(
        UTILIZATION_FEATURES
    )

    # Verify columns exist
    missing_columns = [
        col
        for col in final_columns
        if col not in merged.columns
    ]

    if missing_columns:

        print()
        print(
            "ERROR: Required segmentation "
            "features are missing:"
        )

        for col in missing_columns:
            print(
                f"  - {col}"
            )

        sys.exit(1)

    training_df = merged[
        final_columns
    ].copy()

    # ========================================================
    # NUMERIC CONVERSION
    # ========================================================

    numeric_features = (
        FINANCIAL_FEATURES
        + QUALITY_FEATURES
        + UTILIZATION_FEATURES
    )

    training_df = convert_numeric(
        training_df,
        numeric_features
    )

    # ========================================================
    # SORT
    # ========================================================

    training_df = training_df.sort_values(
        [
            "performance_year",
            "ACO_ID"
        ]
    ).reset_index(drop=True)

    return training_df


# ============================================================
# VALIDATE TRAINING DATASET
# ============================================================

def validate_training_dataset(df):

    print()
    print("=" * 70)
    print("STEP 8A — FINAL TRAINING DATASET VALIDATION")
    print("=" * 70)

    print()
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
        f"{sorted(df['performance_year'].unique())}"
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

    # ========================================================
    # YEAR-WISE COUNTS
    # ========================================================

    print()
    print("-" * 70)
    print("YEAR-WISE TRAINING ROWS")
    print("-" * 70)

    year_counts = (
        df["performance_year"]
        .value_counts()
        .sort_index()
    )

    for year, count in year_counts.items():

        print(
            f"{year}: {count}"
        )

    # ========================================================
    # MISSING VALUES
    # ========================================================

    print()
    print("-" * 70)
    print("MISSING VALUES")
    print("-" * 70)

    missing = df.isna().sum()

    missing = missing[
        missing > 0
    ].sort_values(
        ascending=False
    )

    if missing.empty:

        print(
            "No missing values."
        )

    else:

        for col, count in missing.items():

            pct = (
                count
                / len(df)
                * 100
            )

            print(
                f"{col:45s} "
                f"{count:5d} "
                f"({pct:.2f}%)"
            )

    # ========================================================
    # INFINITE VALUES
    # ========================================================

    print()
    print("-" * 70)
    print("INFINITE VALUE CHECK")
    print("-" * 70)

    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    infinite_count = np.isinf(
        numeric_df.to_numpy()
    ).sum()

    print(
        f"Infinite numeric values: "
        f"{infinite_count}"
    )

    if infinite_count > 0:

        print(
            "WARNING: Infinite values detected."
        )

    # ========================================================
    # FEATURE LIST
    # ========================================================

    print()
    print("-" * 70)
    print("FINAL SEGMENTATION FEATURES")
    print("-" * 70)

    feature_columns = [
        col
        for col in df.columns
        if col not in [
            "ACO_ID",
            "performance_year"
        ]
    ]

    for i, col in enumerate(
        feature_columns,
        start=1
    ):

        print(
            f"{i:02d}. {col}"
        )

    # ========================================================
    # PREVIEW
    # ========================================================

    print()
    print("-" * 70)
    print("DATASET PREVIEW")
    print("-" * 70)

    print(
        df.head(10).to_string(
            index=False
        )
    )


# ============================================================
# SAVE LOCAL DATASET
# ============================================================

def save_dataset(df):

    print()
    print("=" * 70)
    print("SAVING SEGMENTATION TRAINING DATASET")
    print("=" * 70)

    output_directory = os.path.dirname(
        OUTPUT_FILE
    )

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("STEP 8A — ACO PERFORMANCE SEGMENTATION DATASET")
    print("=" * 70)

    # ========================================================
    # UTILIZATION COLUMNS
    # ========================================================

    utilization_columns = [
        "ACO_ID",
        "performance_year",
        "N_AB",
    ]

    utilization_columns.extend(
        UTILIZATION_FEATURES
    )

    # Remove duplicates while preserving order
    utilization_columns = list(
        dict.fromkeys(
            utilization_columns
        )
    )

    # ========================================================
    # FINANCIAL COLUMNS
    # ========================================================

    financial_columns = [
        "ACO_ID",
        "feature_year",
        "target_year",
    ]

    financial_columns.extend(
        FINANCIAL_FEATURES
    )

    financial_columns = list(
        dict.fromkeys(
            financial_columns
        )
    )

    # ========================================================
    # QUALITY COLUMNS
    # ========================================================

    quality_columns = [
        "aco_id",
        "performance_year",
    ]

    quality_columns.extend(
        QUALITY_FEATURES
    )

    quality_columns = list(
        dict.fromkeys(
            quality_columns
        )
    )

    # ========================================================
    # FETCH
    # ========================================================

    utilization_df = fetch_all_rows(
        UTILIZATION_TABLE,
        utilization_columns
    )

    financial_df = fetch_all_rows(
        FINANCIAL_TABLE,
        financial_columns
    )

    quality_df = fetch_all_rows(
        QUALITY_TABLE,
        quality_columns
    )

    # ========================================================
    # CHECK EMPTY
    # ========================================================

    if utilization_df.empty:

        print(
            "ERROR: Utilization table returned no rows."
        )

        sys.exit(1)

    if financial_df.empty:

        print(
            "ERROR: Financial table returned no rows."
        )

        sys.exit(1)

    if quality_df.empty:

        print(
            "ERROR: Quality table returned no rows."
        )

        sys.exit(1)

    # ========================================================
    # BUILD
    # ========================================================

    training_df = build_training_dataset(
        utilization_df,
        financial_df,
        quality_df
    )

    # ========================================================
    # VALIDATE
    # ========================================================

    validate_training_dataset(
        training_df
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_dataset(
        training_df
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 8A COMPLETE")
    print("=" * 70)

    print()
    print(
        "No Supabase data was modified."
    )

    print(
        f"Training dataset created: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Training rows: "
        f"{len(training_df)}"
    )

    print()
    print(
        "READY FOR STEP 8B — FEATURE PREPROCESSING"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()