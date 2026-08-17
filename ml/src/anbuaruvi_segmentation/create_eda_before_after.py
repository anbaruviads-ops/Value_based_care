import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from supabase import create_client, Client
from sklearn.preprocessing import StandardScaler


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


# ============================================================
# PROJECT PATHS
# ============================================================

# This file is assumed to be:
# ACO_Performance_Segmentation/src/eda_before_after.py

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

EVALUATION_DIR = os.path.join(
    BASE_DIR,
    "evaluation"
)

CHART_DIR = os.path.join(
    EVALUATION_DIR,
    "charts"
)

os.makedirs(
    CHART_DIR,
    exist_ok=True
)

OUTPUT_PATH = os.path.join(
    CHART_DIR,
    "11_eda_before_after_preprocessing.png"
)


# ============================================================
# SUPABASE TABLE
# ============================================================

SOURCE_TABLE = "aco_segmentation_ml_features"

PAGE_SIZE = 1000


# ============================================================
# EXACT 15 FEATURES USED BY STEP 8C
# ============================================================

FEATURES = [

    # --------------------------------------------------------
    # Financial performance
    # --------------------------------------------------------

    "SavingsLossPct",
    "ExpenditureVariancePct",
    "FinancialGap",
    "PMPM",

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    "quality_score",

    # --------------------------------------------------------
    # Utilization
    # --------------------------------------------------------

    "utilization_score",
    "ed_visits_per_beneficiary",
    "admissions_per_beneficiary",
    "advanced_imaging_per_beneficiary",
    "em_visit_intensity",

    # --------------------------------------------------------
    # Utilization change
    # --------------------------------------------------------

    "ed_utilization_change_yoy",
    "admission_change_yoy",
    "em_utilization_change_yoy",
    "advanced_imaging_change_yoy",

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    "average_available_risk_score",
]


# ============================================================
# FETCH ALL ROWS
# ============================================================

def fetch_all_rows(table_name):

    print()
    print("=" * 70)
    print("FETCHING SOURCE DATA")
    print("=" * 70)

    rows_all = []

    offset = 0

    while True:

        try:

            response = (
                supabase
                .table(table_name)
                .select("*")
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

        rows_all.extend(rows)

        print(
            f"Retrieved {len(rows_all)} rows"
        )

        if len(rows) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

    print()
    print(
        f"Total rows retrieved: {len(rows_all)}"
    )

    return pd.DataFrame(rows_all)


# ============================================================
# DATASET STATISTICS
# ============================================================

def calculate_dataset_statistics(df):

    rows = len(df)

    columns = len(df.columns)

    missing_cells = int(
        df.isna().sum().sum()
    )

    total_cells = rows * columns

    if total_cells > 0:

        missing_pct = (
            missing_cells
            / total_cells
        ) * 100

    else:

        missing_pct = 0.0

    duplicate_rows = int(
        df.duplicated().sum()
    )

    if rows > 0:

        duplicate_pct = (
            duplicate_rows
            / rows
        ) * 100

    else:

        duplicate_pct = 0.0

    memory_bytes = df.memory_usage(
        deep=True
    ).sum()

    memory_mb = (
        memory_bytes
        / (1024 ** 2)
    )

    if rows > 0:

        avg_record_kb = (
            memory_bytes
            / rows
            / 1024
        )

    else:

        avg_record_kb = 0.0

    return {

        "variables": columns,

        "observations": rows,

        "missing_cells": missing_cells,

        "missing_pct": missing_pct,

        "duplicate_rows": duplicate_rows,

        "duplicate_pct": duplicate_pct,

        "memory_mb": memory_mb,

        "avg_record_kb": avg_record_kb
    }


# ============================================================
# PRINT STATISTICS
# ============================================================

def print_statistics(
    title,
    stats
):

    print()
    print(title)

    print(
        f"variables: "
        f"{stats['variables']}"
    )

    print(
        f"observations: "
        f"{stats['observations']}"
    )

    print(
        f"missing_cells: "
        f"{stats['missing_cells']}"
    )

    print(
        f"missing_pct: "
        f"{stats['missing_pct']:.4f}"
    )

    print(
        f"duplicate_rows: "
        f"{stats['duplicate_rows']}"
    )

    print(
        f"duplicate_pct: "
        f"{stats['duplicate_pct']:.4f}"
    )

    print(
        f"memory_mb: "
        f"{stats['memory_mb']:.6f}"
    )

    print(
        f"avg_record_kb: "
        f"{stats['avg_record_kb']:.6f}"
    )


# ============================================================
# PREPROCESS EXACTLY LIKE STEP 8C
# ============================================================

def preprocess_features(df):

    print()
    print("=" * 70)
    print("PREPROCESSING 15 ML FEATURES")
    print("=" * 70)

    # --------------------------------------------------------
    # Check required features
    # --------------------------------------------------------

    missing_features = [
        col
        for col in FEATURES
        if col not in df.columns
    ]

    if missing_features:

        print()
        print(
            "ERROR: Missing required ML features:"
        )

        for col in missing_features:

            print(
                f"  - {col}"
            )

        sys.exit(1)

    # --------------------------------------------------------
    # Copy only the 15 ML features
    # --------------------------------------------------------

    X = df[
        FEATURES
    ].copy()

    # --------------------------------------------------------
    # Convert to numeric
    # --------------------------------------------------------

    for col in FEATURES:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Missing values BEFORE imputation
    # --------------------------------------------------------

    print()
    print(
        "Missing values in 15 ML features:"
    )

    total_missing = 0

    for col in FEATURES:

        count = int(
            X[col].isna().sum()
        )

        if count > 0:

            print(
                f"{col}: {count}"
            )

        total_missing += count

    print()
    print(
        f"Total missing values: "
        f"{total_missing}"
    )

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Median imputation
    # --------------------------------------------------------

    for col in FEATURES:

        median_value = X[col].median()

        if pd.isna(median_value):

            print()
            print(
                f"ERROR: {col} contains no usable values."
            )

            sys.exit(1)

        X[col] = X[col].fillna(
            median_value
        )

    # --------------------------------------------------------
    # Verify no missing values
    # --------------------------------------------------------

    missing_after_imputation = int(
        X.isna().sum().sum()
    )

    print()
    print(
        "After imputation:"
    )

    print(
        f"Missing cells: "
        f"{missing_after_imputation}"
    )

    # --------------------------------------------------------
    # StandardScaler
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    # Convert back to DataFrame
    # so statistics can be calculated
    # --------------------------------------------------------

    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=FEATURES,
        index=X.index
    )

    print(
        "StandardScaler applied successfully."
    )

    return X, X_scaled_df


# ============================================================
# CREATE EDA FIGURE
# ============================================================

def create_eda_figure(
    before_stats,
    after_stats,
    X_before,
    X_after
):

    print()
    print("=" * 70)
    print("CREATING EDA FIGURE")
    print("=" * 70)

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(16, 10)
    )

    fig.patch.set_alpha(0)

    # ========================================================
    # MAIN TITLE
    # ========================================================

    fig.suptitle(
        "ACO SEGMENTATION — EDA ANALYSIS",
        fontsize=26,
        fontweight="bold",
        y=0.96
    )

    # ========================================================
    # BEFORE PROCESSING PANEL
    # ========================================================

    ax_before = fig.add_axes(
        [0.05, 0.48, 0.42, 0.38]
    )

    ax_before.axis("off")

    before_text = (
        "BEFORE PREPROCESSING\n\n"

        f"Number of variables       "
        f"{before_stats['variables']}\n"

        f"Number of observations    "
        f"{before_stats['observations']}\n"

        f"Missing cells             "
        f"{before_stats['missing_cells']:,}\n"

        f"Missing cells (%)         "
        f"{before_stats['missing_pct']:.2f}%\n"

        f"Duplicate rows            "
        f"{before_stats['duplicate_rows']}\n"

        f"Duplicate rows (%)        "
        f"{before_stats['duplicate_pct']:.2f}%\n"

        f"Memory usage              "
        f"{before_stats['memory_mb']:.3f} MB\n"

        f"Average record size       "
        f"{before_stats['avg_record_kb']:.3f} KB"
    )

    ax_before.text(
        0.02,
        0.95,
        before_text,
        va="top",
        ha="left",
        fontsize=13,
        linespacing=1.6,
        bbox=dict(
            boxstyle="round,pad=0.8",
            alpha=0.12
        )
    )

    # ========================================================
    # BEFORE MISSING VALUES BAR
    # ========================================================

    ax_missing = fig.add_axes(
        [0.55, 0.53, 0.38, 0.30]
    )

    missing_counts = (
        X_before
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing_counts = (
        missing_counts[
            missing_counts > 0
        ]
    )

    if len(missing_counts) > 0:

        ax_missing.bar(
            missing_counts.index,
            missing_counts.values
        )

        ax_missing.set_title(
            "Missing Values by Feature",
            fontsize=15,
            fontweight="bold"
        )

        ax_missing.set_ylabel(
            "Missing Count"
        )

        ax_missing.tick_params(
            axis="x",
            rotation=90
        )

    else:

        ax_missing.text(
            0.5,
            0.5,
            "No missing values",
            ha="center",
            va="center",
            fontsize=14
        )

        ax_missing.set_title(
            "Missing Values by Feature"
        )

    # ========================================================
    # AFTER PROCESSING PANEL
    # ========================================================

    ax_after = fig.add_axes(
        [0.05, 0.06, 0.42, 0.32]
    )

    ax_after.axis("off")

    after_text = (
        "AFTER PREPROCESSING\n\n"

        f"Number of variables       "
        f"{after_stats['variables']}\n"

        f"Number of observations    "
        f"{after_stats['observations']}\n"

        f"Missing cells             "
        f"{after_stats['missing_cells']:,}\n"

        f"Missing cells (%)         "
        f"{after_stats['missing_pct']:.2f}%\n"

        f"Duplicate rows            "
        f"{after_stats['duplicate_rows']}\n"

        f"Duplicate rows (%)        "
        f"{after_stats['duplicate_pct']:.2f}%\n"

        f"Memory usage              "
        f"{after_stats['memory_mb']:.3f} MB\n"

        f"Average record size       "
        f"{after_stats['avg_record_kb']:.3f} KB"
    )

    ax_after.text(
        0.02,
        0.95,
        after_text,
        va="top",
        ha="left",
        fontsize=13,
        linespacing=1.6,
        bbox=dict(
            boxstyle="round,pad=0.8",
            alpha=0.12
        )
    )

    # ========================================================
    # AFTER DISTRIBUTION
    # ========================================================

    ax_after_dist = fig.add_axes(
        [0.55, 0.08, 0.38, 0.27]
    )

    # Use standardized values from all 15 features
    values = X_after.values.flatten()

    ax_after_dist.hist(
        values,
        bins=30
    )

    ax_after_dist.set_title(
        "Distribution After Standardization",
        fontsize=15,
        fontweight="bold"
    )

    ax_after_dist.set_xlabel(
        "Standardized Feature Value"
    )

    ax_after_dist.set_ylabel(
        "Frequency"
    )

    # ========================================================
    # FOOTER
    # ========================================================

    fig.text(
        0.5,
        0.015,
        (
            "Preprocessing: numeric conversion → "
            "infinite-value handling → median imputation → "
            "StandardScaler"
        ),
        ha="center",
        fontsize=10
    )

    # ========================================================
    # SAVE
    # ========================================================

    plt.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        f"EDA chart saved to:"
    )

    print(
        OUTPUT_PATH
    )

    return OUTPUT_PATH


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "ACO SEGMENTATION — BEFORE / AFTER EDA"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Fetch source data
    # --------------------------------------------------------

    df = fetch_all_rows(
        SOURCE_TABLE
    )

    if df.empty:

        print()
        print(
            "ERROR: Source table is empty."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # BEFORE statistics
    # --------------------------------------------------------

    before_stats = calculate_dataset_statistics(
        df
    )

    print_statistics(
        "BEFORE PREPROCESSING",
        before_stats
    )

    # --------------------------------------------------------
    # Keep a copy of raw 15 features
    # for the BEFORE missing-value chart
    # --------------------------------------------------------

    missing_feature_check = [
        col
        for col in FEATURES
        if col not in df.columns
    ]

    if missing_feature_check:

        print()
        print(
            "ERROR: Required ML features missing:"
        )

        for col in missing_feature_check:

            print(
                f"  - {col}"
            )

        sys.exit(1)

    X_before = df[
        FEATURES
    ].copy()

    for col in FEATURES:

        X_before[col] = pd.to_numeric(
            X_before[col],
            errors="coerce"
        )

    X_before = X_before.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    X_imputed, X_scaled = preprocess_features(
        df
    )

    # --------------------------------------------------------
    # AFTER statistics
    #
    # The final ML dataset contains exactly
    # the 15 features used by K-Means.
    # --------------------------------------------------------

    after_stats = calculate_dataset_statistics(
        X_scaled
    )

    print_statistics(
        "AFTER PREPROCESSING",
        after_stats
    )

    # --------------------------------------------------------
    # Create chart
    # --------------------------------------------------------

    output = create_eda_figure(
        before_stats,
        after_stats,
        X_before,
        X_scaled
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "EDA ANALYSIS COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        f"Output: {output}"
    )

    print()
    print(
        "This analysis is READ-ONLY."
    )

    print(
        "No Supabase data was modified."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()