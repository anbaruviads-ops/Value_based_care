import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/segmentation_training_dataset.csv"

OUTPUT_FEATURE_FILE = (
    "data/segmentation_features_scaled.csv"
)

OUTPUT_METADATA_FILE = (
    "data/segmentation_feature_metadata.csv"
)

OUTPUT_SCALER_FILE = (
    "models/segmentation_scaler.joblib"
)


# ============================================================
# IDENTIFIER COLUMNS
# ============================================================

ID_COLUMNS = [
    "ACO_ID",
    "performance_year"
]


# ============================================================
# INITIAL FEATURE SET
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


INITIAL_FEATURES = (
    FINANCIAL_FEATURES
    + QUALITY_FEATURES
    + UTILIZATION_FEATURES
)


# ============================================================
# FEATURES TO REMOVE FROM K-MEANS
# ============================================================

# These are redundant because they duplicate information
# already represented by another selected feature.

REDUNDANT_FEATURES = [
    # BenchmarkPMPM and PMPM together with FinancialGap
    # create overlapping financial information.
    "BenchmarkPMPM",

    # These rate variables strongly overlap with their
    # corresponding utilization-per-beneficiary variables.
    "advanced_imaging_per_beneficiary",

    # Utilization score already summarizes several utilization
    # dimensions. We retain it as the high-level utilization
    # indicator and remove some direct components.
    "snf_admissions_per_beneficiary",
]


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print()
    print("=" * 70)
    print("LOADING STEP 8A DATASET")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):

        print()
        print(
            f"ERROR: Input file not found:"
            f"\n{INPUT_FILE}"
        )

        print()
        print(
            "Run Step 8A first:"
        )

        print(
            "python "
            "src/build_segmentation_training_dataset.py"
        )

        sys.exit(1)

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Rows loaded: {len(df)}"
    )

    print(
        f"Columns loaded: {len(df.columns)}"
    )

    return df


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input(df):

    print()
    print("=" * 70)
    print("VALIDATING INPUT DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Required ID columns
    # --------------------------------------------------------

    for col in ID_COLUMNS:

        if col not in df.columns:

            print(
                f"ERROR: Missing ID column: {col}"
            )

            sys.exit(1)

    # --------------------------------------------------------
    # Required features
    # --------------------------------------------------------

    missing_features = [
        col
        for col in INITIAL_FEATURES
        if col not in df.columns
    ]

    if missing_features:

        print()
        print(
            "ERROR: Required features missing:"
        )

        for col in missing_features:
            print(
                f"  - {col}"
            )

        sys.exit(1)

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    duplicates = df.duplicated(
        subset=ID_COLUMNS
    ).sum()

    print(
        f"Duplicate ACO-Year rows: "
        f"{duplicates}"
    )

    if duplicates > 0:

        print(
            "ERROR: Duplicate ACO-Year rows found."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Year check
    # --------------------------------------------------------

    print(
        "Years:",
        sorted(
            df["performance_year"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    # --------------------------------------------------------
    # ACO count
    # --------------------------------------------------------

    print(
        f"Unique ACOs: "
        f"{df['ACO_ID'].nunique()}"
    )


# ============================================================
# CONVERT FEATURES TO NUMERIC
# ============================================================

def convert_features_to_numeric(df):

    print()
    print("=" * 70)
    print("CONVERTING FEATURES TO NUMERIC")
    print("=" * 70)

    for col in INITIAL_FEATURES:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df


# ============================================================
# INFINITE VALUE CHECK
# ============================================================

def handle_infinite_values(df):

    print()
    print("=" * 70)
    print("CHECKING INFINITE VALUES")
    print("=" * 70)

    numeric_df = df[
        INITIAL_FEATURES
    ]

    infinite_mask = np.isinf(
        numeric_df.to_numpy()
    )

    infinite_count = infinite_mask.sum()

    print(
        f"Infinite values found: "
        f"{infinite_count}"
    )

    if infinite_count > 0:

        print(
            "Replacing infinite values with NaN."
        )

        df[INITIAL_FEATURES] = (
            df[INITIAL_FEATURES]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

    return df


# ============================================================
# MISSING VALUE REPORT
# ============================================================

def missing_value_report(df):

    print()
    print("=" * 70)
    print("MISSING VALUE REPORT")
    print("=" * 70)

    report = []

    for col in INITIAL_FEATURES:

        missing = int(
            df[col].isna().sum()
        )

        percentage = (
            missing
            / len(df)
            * 100
        )

        report.append(
            {
                "feature": col,
                "missing_count": missing,
                "missing_percentage": percentage
            }
        )

        print(
            f"{col:45s} "
            f"{missing:5d} "
            f"({percentage:.2f}%)"
        )

    return pd.DataFrame(report)


# ============================================================
# IMPUTE MISSING VALUES
# ============================================================

def impute_missing_values(df):

    print()
    print("=" * 70)
    print("HANDLING MISSING VALUES")
    print("=" * 70)

    # Median imputation is safer than mean for healthcare
    # utilization and financial variables because these data
    # can contain extreme values.

    for col in INITIAL_FEATURES:

        missing_count = (
            df[col]
            .isna()
            .sum()
        )

        if missing_count > 0:

            median_value = (
                df[col]
                .median()
            )

            if pd.isna(median_value):

                print()
                print(
                    f"ERROR: Entire feature is missing: "
                    f"{col}"
                )

                sys.exit(1)

            df[col] = (
                df[col]
                .fillna(median_value)
            )

            print(
                f"{col}: "
                f"filled {missing_count} "
                f"values using median "
                f"{median_value:.6f}"
            )

    return df


# ============================================================
# REMOVE REDUNDANT FEATURES
# ============================================================

def select_features(df):

    print()
    print("=" * 70)
    print("FEATURE SELECTION")
    print("=" * 70)

    selected_features = [
        col
        for col in INITIAL_FEATURES
        if col not in REDUNDANT_FEATURES
    ]

    print()
    print(
        f"Initial features: "
        f"{len(INITIAL_FEATURES)}"
    )

    print(
        f"Removed redundant features: "
        f"{len(REDUNDANT_FEATURES)}"
    )

    for col in REDUNDANT_FEATURES:

        print(
            f"  - {col}"
        )

    print()
    print(
        f"Final features: "
        f"{len(selected_features)}"
    )

    print()
    print(
        "Selected features:"
    )

    for i, col in enumerate(
        selected_features,
        start=1
    ):

        print(
            f"{i:02d}. {col}"
        )

    return selected_features


# ============================================================
# CORRELATION CHECK
# ============================================================

def correlation_check(
    df,
    selected_features
):

    print()
    print("=" * 70)
    print("CORRELATION CHECK")
    print("=" * 70)

    correlation_matrix = (
        df[selected_features]
        .corr()
        .abs()
    )

    high_correlations = []

    for i in range(
        len(selected_features)
    ):

        for j in range(
            i + 1,
            len(selected_features)
        ):

            feature_a = (
                selected_features[i]
            )

            feature_b = (
                selected_features[j]
            )

            correlation = (
                correlation_matrix
                .loc[
                    feature_a,
                    feature_b
                ]
            )

            if correlation >= 0.90:

                high_correlations.append(
                    {
                        "feature_1": feature_a,
                        "feature_2": feature_b,
                        "absolute_correlation": correlation
                    }
                )

    if not high_correlations:

        print(
            "No feature pairs with "
            "|correlation| >= 0.90."
        )

    else:

        print(
            "High-correlation pairs:"
        )

        for item in high_correlations:

            print(
                f"{item['feature_1']} "
                f"<-> "
                f"{item['feature_2']} "
                f"= "
                f"{item['absolute_correlation']:.4f}"
            )

    return pd.DataFrame(
        high_correlations
    )


# ============================================================
# STANDARDIZE FEATURES
# ============================================================

def scale_features(
    df,
    selected_features
):

    print()
    print("=" * 70)
    print("STANDARDIZING FEATURES")
    print("=" * 70)

    scaler = StandardScaler()

    X = df[
        selected_features
    ].copy()

    X_scaled = scaler.fit_transform(
        X
    )

    scaled_df = pd.DataFrame(
        X_scaled,
        columns=selected_features,
        index=df.index
    )

    print(
        f"Rows scaled: "
        f"{len(scaled_df)}"
    )

    print(
        f"Features scaled: "
        f"{len(selected_features)}"
    )

    # --------------------------------------------------------
    # Verify scaling
    # --------------------------------------------------------

    print()
    print(
        "Scaled feature means:"
    )

    print(
        scaled_df.mean()
        .round(6)
        .to_string()
    )

    print()
    print(
        "Scaled feature standard deviations:"
    )

    print(
        scaled_df.std(
            ddof=0
        )
        .round(6)
        .to_string()
    )

    return scaled_df, scaler


# ============================================================
# SAVE FEATURE METADATA
# ============================================================

def save_metadata(
    selected_features,
    missing_report,
    correlation_report
):

    os.makedirs(
        "data",
        exist_ok=True
    )

    metadata_rows = []

    for feature in selected_features:

        missing_row = (
            missing_report[
                missing_report["feature"]
                == feature
            ]
        )

        if not missing_row.empty:

            missing_count = int(
                missing_row.iloc[0][
                    "missing_count"
                ]
            )

            missing_percentage = float(
                missing_row.iloc[0][
                    "missing_percentage"
                ]
            )

        else:

            missing_count = 0
            missing_percentage = 0.0

        metadata_rows.append(
            {
                "feature": feature,
                "missing_count": missing_count,
                "missing_percentage": missing_percentage,
                "selected_for_model": True
            }
        )

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    metadata_df.to_csv(
        OUTPUT_METADATA_FILE,
        index=False
    )

    print()
    print(
        f"Feature metadata saved: "
        f"{OUTPUT_METADATA_FILE}"
    )


# ============================================================
# SAVE SCALER
# ============================================================

def save_scaler(scaler):

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        scaler,
        OUTPUT_SCALER_FILE
    )

    print(
        f"Scaler saved: "
        f"{OUTPUT_SCALER_FILE}"
    )


# ============================================================
# SAVE SCALED DATASET
# ============================================================

def save_scaled_dataset(
    df,
    scaled_df
):

    output_df = pd.concat(
        [
            df[
                ID_COLUMNS
            ].reset_index(drop=True),

            scaled_df.reset_index(drop=True)
        ],
        axis=1
    )

    output_df.to_csv(
        OUTPUT_FEATURE_FILE,
        index=False
    )

    print()
    print(
        f"Scaled feature dataset saved:"
    )

    print(
        OUTPUT_FEATURE_FILE
    )

    print(
        f"Rows: "
        f"{len(output_df)}"
    )

    print(
        f"Columns: "
        f"{len(output_df.columns)}"
    )

    return output_df


# ============================================================
# FINAL VALIDATION
# ============================================================

def final_validation(
    output_df,
    selected_features
):

    print()
    print("=" * 70)
    print("FINAL STEP 8B VALIDATION")
    print("=" * 70)

    print(
        f"Rows: "
        f"{len(output_df)}"
    )

    print(
        f"Columns: "
        f"{len(output_df.columns)}"
    )

    print(
        f"Unique ACOs: "
        f"{output_df['ACO_ID'].nunique()}"
    )

    print(
        f"Years: "
        f"{sorted(output_df['performance_year'].unique())}"
    )

    duplicates = output_df.duplicated(
        subset=ID_COLUMNS
    ).sum()

    print(
        f"Duplicate ACO-Year rows: "
        f"{duplicates}"
    )

    # --------------------------------------------------------
    # Missing check
    # --------------------------------------------------------

    missing_count = (
        output_df[
            selected_features
        ]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing feature values: "
        f"{missing_count}"
    )

    # --------------------------------------------------------
    # Infinite check
    # --------------------------------------------------------

    infinite_count = np.isinf(
        output_df[
            selected_features
        ]
        .to_numpy()
    ).sum()

    print(
        f"Infinite feature values: "
        f"{infinite_count}"
    )

    # --------------------------------------------------------
    # Standardization check
    # --------------------------------------------------------

    means = (
        output_df[
            selected_features
        ]
        .mean()
        .abs()
    )

    max_mean = means.max()

    print(
        f"Maximum absolute scaled mean: "
        f"{max_mean:.8f}"
    )

    if max_mean > 0.0001:

        print(
            "WARNING: Some features may not "
            "be properly standardized."
        )

    else:

        print(
            "Scaling validation: PASS"
        )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    if (
        duplicates == 0
        and missing_count == 0
        and infinite_count == 0
    ):

        print()
        print(
            "STEP 8B VALIDATION: PASS"
        )

    else:

        print()
        print(
            "STEP 8B VALIDATION: FAILED"
        )

        sys.exit(1)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("STEP 8B — SEGMENTATION FEATURE PREPROCESSING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_input(
        df
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    df = convert_features_to_numeric(
        df
    )

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    df = handle_infinite_values(
        df
    )

    # --------------------------------------------------------
    # Missing report BEFORE imputation
    # --------------------------------------------------------

    missing_report = missing_value_report(
        df
    )

    # --------------------------------------------------------
    # Impute
    # --------------------------------------------------------

    df = impute_missing_values(
        df
    )

    # --------------------------------------------------------
    # Select features
    # --------------------------------------------------------

    selected_features = select_features(
        df
    )

    # --------------------------------------------------------
    # Correlation check
    # --------------------------------------------------------

    correlation_report = correlation_check(
        df,
        selected_features
    )

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    scaled_df, scaler = scale_features(
        df,
        selected_features
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    save_metadata(
        selected_features,
        missing_report,
        correlation_report
    )

    # --------------------------------------------------------
    # Save scaler
    # --------------------------------------------------------

    save_scaler(
        scaler
    )

    # --------------------------------------------------------
    # Save scaled dataset
    # --------------------------------------------------------

    output_df = save_scaled_dataset(
        df,
        scaled_df
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    final_validation(
        output_df,
        selected_features
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 8B COMPLETE")
    print("=" * 70)

    print()
    print(
        "No Supabase data was modified."
    )

    print()
    print(
        "Files created:"
    )

    print(
        f"  1. {OUTPUT_FEATURE_FILE}"
    )

    print(
        f"  2. {OUTPUT_METADATA_FILE}"
    )

    print(
        f"  3. {OUTPUT_SCALER_FILE}"
    )

    print()
    print(
        "READY FOR STEP 8C — K-MEANS MODEL TRAINING"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()