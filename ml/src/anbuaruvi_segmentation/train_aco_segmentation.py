import os
import sys
import pickle
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from supabase import create_client, Client

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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

SOURCE_TABLE = "aco_segmentation_ml_features"
TARGET_TABLE = "aco_segmentation_results"

PAGE_SIZE = 1000
N_CLUSTERS = 3

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "aco_segmentation_kmeans.pkl"
)


# ============================================================
# SEGMENTATION FEATURES
# ============================================================

FEATURES = [
    # Financial performance
    "SavingsLossPct",
    "ExpenditureVariancePct",
    "FinancialGap",
    "PMPM",

    # Quality
    "quality_score",

    # Utilization
    "utilization_score",
    "ed_visits_per_beneficiary",
    "admissions_per_beneficiary",
    "advanced_imaging_per_beneficiary",
    "em_visit_intensity",

    # Utilization change
    "ed_utilization_change_yoy",
    "admission_change_yoy",
    "em_utilization_change_yoy",
    "advanced_imaging_change_yoy",

    # Risk
    "average_available_risk_score",
]


# ============================================================
# FETCH ALL ROWS
# ============================================================

def fetch_all_rows(table_name):

    print()
    print("=" * 70)
    print(f"FETCHING: {table_name}")
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

    print(
        f"Total rows retrieved: {len(rows_all)}"
    )

    return pd.DataFrame(rows_all)


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    print()
    print("=" * 70)
    print("PREPARING SEGMENTATION FEATURES")
    print("=" * 70)

    missing_features = [
        col
        for col in FEATURES
        if col not in df.columns
    ]

    if missing_features:

        print("ERROR: Missing required features:")

        for col in missing_features:
            print(f"  - {col}")

        sys.exit(1)

    # Convert ML columns to numeric
    for col in FEATURES:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    print()
    print("Missing values before imputation:")

    for col in FEATURES:

        count = df[col].isna().sum()

        if count > 0:

            print(
                f"{col}: {count}"
            )

    # --------------------------------------------------------
    # Median imputation
    # --------------------------------------------------------

    X = df[FEATURES].copy()

    for col in FEATURES:

        median_value = X[col].median()

        if pd.isna(median_value):

            print(
                f"ERROR: Feature {col} contains no usable values."
            )

            sys.exit(1)

        X[col] = X[col].fillna(
            median_value
        )

    # --------------------------------------------------------
    # Remove infinite values
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    for col in FEATURES:

        X[col] = X[col].fillna(
            X[col].median()
        )

    print()
    print(
        f"Rows used for training: {len(X)}"
    )

    print(
        f"Features used: {len(FEATURES)}"
    )

    return X


# ============================================================
# TRAIN K-MEANS
# ============================================================

def train_model(X):

    print()
    print("=" * 70)
    print("TRAINING K-MEANS SEGMENTATION MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------------
    # Train K-Means
    # --------------------------------------------------------

    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=20
    )

    cluster_ids = model.fit_predict(
        X_scaled
    )

    # --------------------------------------------------------
    # Silhouette score
    # --------------------------------------------------------

    silhouette = silhouette_score(
        X_scaled,
        cluster_ids
    )

    print()
    print(
        f"Number of clusters: {N_CLUSTERS}"
    )

    print(
        f"Silhouette score: {silhouette:.4f}"
    )

    print()
    print("Cluster counts:")

    counts = pd.Series(
        cluster_ids
    ).value_counts().sort_index()

    for cluster_id, count in counts.items():

        print(
            f"Cluster {cluster_id}: {count}"
        )

    return model, scaler, cluster_ids, silhouette


# ============================================================
# BUILD CLUSTER PROFILES
# ============================================================

def build_cluster_profiles(
    df,
    X,
    cluster_ids
):

    print()
    print("=" * 70)
    print("BUILDING CLUSTER PROFILES")
    print("=" * 70)

    profile = X.copy()

    profile["cluster_id"] = cluster_ids

    # Add original business metrics where available
    extra_columns = [
        "ACO_ID",
        "performance_year",
    ]

    for col in extra_columns:

        if col in df.columns:
            profile[col] = df[col].values

    cluster_profile = (
        profile
        .groupby("cluster_id")[FEATURES]
        .mean()
    )

    print()
    print(
        cluster_profile.round(3)
    )

    return cluster_profile


# ============================================================
# ASSIGN BUSINESS SEGMENTS
# ============================================================

def assign_business_segments(
    cluster_profile
):

    print()
    print("=" * 70)
    print("INTERPRETING CLUSTERS")
    print("=" * 70)

    # --------------------------------------------------------
    # Create a business-oriented performance index.
    #
    # Higher:
    #   Savings
    #   Quality
    #
    # Lower:
    #   Expenditure variance
    #   Utilization
    # --------------------------------------------------------

    score = pd.DataFrame(
        index=cluster_profile.index
    )

    def normalize(series):

        minimum = series.min()
        maximum = series.max()

        if maximum == minimum:

            return pd.Series(
                0.5,
                index=series.index
            )

        return (
            (series - minimum)
            / (maximum - minimum)
        )

    savings = normalize(
        cluster_profile["SavingsLossPct"]
    )

    quality = normalize(
        cluster_profile["quality_score"]
    )

    utilization = normalize(
        cluster_profile["utilization_score"]
    )

    expenditure = normalize(
        cluster_profile["ExpenditureVariancePct"]
    )

    score["performance_index"] = (
        0.35 * savings
        + 0.30 * quality
        + 0.20 * (1 - utilization)
        + 0.15 * (1 - expenditure)
    )

    print()
    print(
        "Cluster performance scores:"
    )

    print(
        score.round(4)
    )

    # Highest performance index
    ordered_clusters = (
        score["performance_index"]
        .sort_values()
        .index
        .tolist()
    )

    segment_map = {}

    if len(ordered_clusters) == 3:

        segment_map[
            ordered_clusters[0]
        ] = "Needs Attention"

        segment_map[
            ordered_clusters[1]
        ] = "Moderate"

        segment_map[
            ordered_clusters[2]
        ] = "High Performing"

    else:

        print(
            "ERROR: Expected exactly 3 clusters."
        )

        sys.exit(1)

    print()
    print("Cluster → Business Segment")

    for cluster_id in sorted(segment_map):

        print(
            f"Cluster {cluster_id} "
            f"→ {segment_map[cluster_id]}"
        )

    return segment_map, score


# ============================================================
# CREATE RESULTS
# ============================================================

def create_results(
    df,
    cluster_ids,
    segment_map,
    performance_scores
):

    result = pd.DataFrame()

    result["ACO_ID"] = df["ACO_ID"]

    # Source table should use performance_year.
    # If the segmentation table uses another name,
    # this fallback handles it.
    if "performance_year" in df.columns:

        result["performance_year"] = (
            pd.to_numeric(
                df["performance_year"],
                errors="coerce"
            )
        )

    elif "target_year" in df.columns:

        result["performance_year"] = (
            pd.to_numeric(
                df["target_year"],
                errors="coerce"
            )
        )

    else:

        print(
            "ERROR: No performance year column found."
        )

        sys.exit(1)

    result["cluster_id"] = (
        cluster_ids.astype(int)
    )

    result["performance_segment"] = [
        segment_map[int(cluster)]
        for cluster in cluster_ids
    ]

    result["performance_index"] = [
        float(
            performance_scores.loc[
                int(cluster),
                "performance_index"
            ]
        )
        for cluster in cluster_ids
    ]

    # Add important model inputs
    for col in [
        "SavingsLossPct",
        "ExpenditureVariancePct",
        "quality_score",
        "utilization_score",
    ]:

        if col in df.columns:

            result[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # Clean invalid values
    result = result.replace(
        [np.inf, -np.inf],
        np.nan
    )

    result = result.where(
        pd.notnull(result),
        None
    )

    return result


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    scaler
):

    print()
    print("=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model_package = {
        "model": model,
        "scaler": scaler,
        "features": FEATURES,
        "n_clusters": N_CLUSTERS,
        "random_state": 42
    }

    with open(
        MODEL_PATH,
        "wb"
    ) as file:

        pickle.dump(
            model_package,
            file
        )

    print(
        f"Model saved to: {MODEL_PATH}"
    )


# ============================================================
# INSERT RESULTS
# ============================================================

def insert_results(result):

    print()
    print("=" * 70)
    print("WRITING SEGMENTATION RESULTS")
    print("=" * 70)

    records = result.to_dict(
        orient="records"
    )

    # Convert NumPy values to Python values
    for record in records:

        for key, value in record.items():

            if isinstance(
                value,
                (np.integer,)
            ):

                record[key] = int(value)

            elif isinstance(
                value,
                (np.floating,)
            ):

                if np.isfinite(value):
                    record[key] = float(value)
                else:
                    record[key] = None

            elif pd.isna(value):

                record[key] = None

    batch_size = 500

    for start in range(
        0,
        len(records),
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
                        "ACO_ID,performance_year"
                    )
                )
                .execute()
            )

        except Exception as e:

            print()
            print(
                "ERROR WRITING RESULTS"
            )

            print(e)

            sys.exit(1)

        print(
            f"Inserted/upserted "
            f"{min(start + batch_size, len(records))}"
            f" / {len(records)}"
        )

    print()
    print(
        "SEGMENTATION RESULTS INSERTED SUCCESSFULLY"
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_results(result):

    print()
    print("=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    print(
        f"Rows: {len(result)}"
    )

    print(
        f"Unique ACOs: "
        f"{result['ACO_ID'].nunique()}"
    )

    print(
        f"Years: "
        f"{sorted(result['performance_year'].dropna().unique())}"
    )

    duplicates = result.duplicated(
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
    print("Business segment distribution:")

    print(
        result[
            "performance_segment"
        ].value_counts()
    )

    print()
    print("Cluster distribution:")

    print(
        result[
            "cluster_id"
        ].value_counts()
        .sort_index()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "STEP 8C — ACO PERFORMANCE SEGMENTATION MODEL"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Fetch
    # --------------------------------------------------------

    df = fetch_all_rows(
        SOURCE_TABLE
    )

    if df.empty:

        print(
            "ERROR: Source table is empty."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Validate ACO-Year uniqueness
    # --------------------------------------------------------

    duplicates = df.duplicated(
        subset=[
            "ACO_ID",
            "performance_year"
        ]
    ).sum()

    print()
    print(
        f"Duplicate ACO-Year rows: {duplicates}"
    )

    if duplicates > 0:

        print(
            "ERROR: Source table contains duplicate ACO-Year rows."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X = prepare_features(
        df
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    (
        model,
        scaler,
        cluster_ids,
        silhouette
    ) = train_model(
        X
    )

    # --------------------------------------------------------
    # Cluster profiles
    # --------------------------------------------------------

    cluster_profile = (
        build_cluster_profiles(
            df,
            X,
            cluster_ids
        )
    )

    # --------------------------------------------------------
    # Business interpretation
    # --------------------------------------------------------

    (
        segment_map,
        performance_scores
    ) = assign_business_segments(
        cluster_profile
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    result = create_results(
        df,
        cluster_ids,
        segment_map,
        performance_scores
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate_results(
        result
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    save_model(
        model,
        scaler
    )

    # --------------------------------------------------------
    # Write results
    # --------------------------------------------------------

    insert_results(
        result
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 8C COMPLETE")
    print("=" * 70)

    print(
        f"Silhouette score: {silhouette:.4f}"
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        f"Results table: {TARGET_TABLE}"
    )

    print()
    print(
        "Source tables were not modified."
    )


if __name__ == "__main__":
    main()