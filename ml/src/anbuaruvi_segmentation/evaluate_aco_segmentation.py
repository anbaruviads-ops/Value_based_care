import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/aco_segmentation_kmeans.pkl"
OUTPUT_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "SavingsLossPct",
    "ExpenditureVariancePct",
    "FinancialGap",
    "PMPM",
    "quality_score",
    "utilization_score",
    "ed_visits_per_beneficiary",
    "admissions_per_beneficiary",
    "advanced_imaging_per_beneficiary",
    "em_visit_intensity",
    "ed_utilization_change_yoy",
    "admission_change_yoy",
    "em_utilization_change_yoy",
    "advanced_imaging_change_yoy",
    "average_available_risk_score"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_training_data():

    print("=" * 70)
    print("LOADING SEGMENTATION DATA")
    print("=" * 70)

    from dotenv import load_dotenv
    from supabase import create_client

    load_dotenv()

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    if not SUPABASE_URL or not SUPABASE_KEY:

        print("ERROR: Supabase credentials missing.")
        sys.exit(1)

    supabase = create_client(
        SUPABASE_URL,
        SUPABASE_KEY
    )

    table = "aco_segmentation_ml_features"

    columns = [
        "ACO_ID",
        "performance_year"
    ] + FEATURES

    all_rows = []

    offset = 0
    page_size = 1000

    while True:

        response = (
            supabase
            .table(table)
            .select(",".join(columns))
            .range(
                offset,
                offset + page_size - 1
            )
            .execute()
        )

        rows = response.data

        if not rows:
            break

        all_rows.extend(rows)

        print(
            f"Retrieved {len(all_rows)} rows"
        )

        if len(rows) < page_size:
            break

        offset += page_size

    df = pd.DataFrame(all_rows)

    print(
        f"Total rows: {len(df)}"
    )

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df, saved_features):

    print()
    print("=" * 70)
    print("PREPARING FEATURES")
    print("=" * 70)

    # --------------------------------------------------------
    # Make sure evaluation uses exactly the same features
    # as training
    # --------------------------------------------------------

    print("Features used by trained model:")

    for feature in saved_features:
        print(f"  - {feature}")

    missing_features = [
        feature
        for feature in saved_features
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing features from Supabase data: "
            + str(missing_features)
        )

    X = df[saved_features].copy()

    # --------------------------------------------------------
    # Convert to numeric
    # --------------------------------------------------------

    for col in saved_features:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    print()
    print("Missing values before imputation:")

    missing = X.isna().sum()

    print(
        missing.loc[lambda x: x > 0]
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Match the training preprocessing
    #
    # Your original script used median imputation.
    # --------------------------------------------------------

    X = X.fillna(
        X.median()
    )

    return X


# ============================================================
# LOAD SAVED MODEL
# ============================================================

def load_model():

    print()
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):

        print(
            f"ERROR: Model not found: {MODEL_PATH}"
        )

        sys.exit(1)

    model_data = joblib.load(
        MODEL_PATH
    )

    print(
        f"Model loaded: {MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Validate saved structure
    # --------------------------------------------------------

    if not isinstance(model_data, dict):

        raise ValueError(
            "Expected saved model to be a dictionary."
        )

    required_keys = [
        "model",
        "scaler",
        "features",
        "n_clusters",
        "random_state"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in model_data
    ]

    if missing_keys:

        raise ValueError(
            "Saved model is missing keys: "
            + str(missing_keys)
        )

    model = model_data["model"]
    scaler = model_data["scaler"]
    saved_features = model_data["features"]

    print()
    print("Saved model components:")

    print(
        f"  KMeans model : {type(model)}"
    )

    print(
        f"  Scaler       : {type(scaler)}"
    )

    print(
        f"  Features     : {len(saved_features)}"
    )

    print(
        f"  K             : {model_data['n_clusters']}"
    )

    print(
        f"  Random state  : {model_data['random_state']}"
    )

    return (
        model,
        scaler,
        saved_features,
        model_data
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    scaler,
    X
):

    print()
    print("=" * 70)
    print("MODEL PERFORMANCE EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Apply the SAME scaler used during training
    # --------------------------------------------------------

    X_scaled = scaler.transform(X)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    cluster_labels = model.predict(
        X_scaled
    )

    unique_clusters = np.unique(
        cluster_labels
    )

    print()
    print(
        f"Clusters detected: "
        f"{unique_clusters.tolist()}"
    )

    # --------------------------------------------------------
    # Silhouette Score
    # --------------------------------------------------------

    silhouette = silhouette_score(
        X_scaled,
        cluster_labels
    )

    # --------------------------------------------------------
    # Calinski-Harabasz Score
    # --------------------------------------------------------

    calinski = calinski_harabasz_score(
        X_scaled,
        cluster_labels
    )

    # --------------------------------------------------------
    # Davies-Bouldin Score
    # --------------------------------------------------------

    davies = davies_bouldin_score(
        X_scaled,
        cluster_labels
    )

    # --------------------------------------------------------
    # Inertia
    # --------------------------------------------------------

    inertia = None

    if hasattr(model, "inertia_"):

        inertia = model.inertia_

    # --------------------------------------------------------
    # Cluster distribution
    # --------------------------------------------------------

    cluster_counts = (
        pd.Series(cluster_labels)
        .value_counts()
        .sort_index()
    )

    print()
    print(
        f"Silhouette Score      : "
        f"{silhouette:.4f}"
    )

    print(
        f"Calinski-Harabasz     : "
        f"{calinski:.4f}"
    )

    print(
        f"Davies-Bouldin Score  : "
        f"{davies:.4f}"
    )

    if inertia is not None:

        print(
            f"Inertia               : "
            f"{inertia:.4f}"
        )

    print()
    print("Cluster distribution:")

    for cluster, count in cluster_counts.items():

        percentage = (
            count /
            len(cluster_labels)
            * 100
        )

        print(
            f"Cluster {cluster}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    # ========================================================
    # METRICS DICTIONARY
    # ========================================================

    metrics = {

        "model": "KMeans",

        "n_clusters": int(
            len(unique_clusters)
        ),

        "silhouette_score": float(
            silhouette
        ),

        "calinski_harabasz_score": float(
            calinski
        ),

        "davies_bouldin_score": float(
            davies
        ),

        "inertia": (
            float(inertia)
            if inertia is not None
            else None
        ),

        "cluster_distribution": {

            str(cluster): int(count)

            for cluster, count
            in cluster_counts.items()

        }
    }

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_path = os.path.join(
        OUTPUT_DIR,
        "segmentation_metrics.json"
    )

    with open(
        metrics_path,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print()
    print(
        f"Metrics saved to: "
        f"{metrics_path}"
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "ACO PERFORMANCE SEGMENTATION — MODEL EVALUATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_training_data()

    # --------------------------------------------------------
    # Load trained model + scaler
    # --------------------------------------------------------

    (
        model,
        scaler,
        saved_features,
        model_data
    ) = load_model()

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X = prepare_features(
        df,
        saved_features
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluate_model(
        model,
        scaler,
        X
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()