import os
import sys
import json
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from supabase import create_client, Client

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("ERROR: SUPABASE_URL or SUPABASE_KEY is missing.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

SOURCE_TABLE = "aco_segmentation_ml_features"
PAGE_SIZE = 1000

KMEANS_K_VALUES = [2, 3, 4]
COMPARISON_K = 3

OUTPUT_DIR = "evaluation"
OUTPUT_CSV = os.path.join(
    OUTPUT_DIR, "clustering_model_comparison.csv"
)

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
    "average_available_risk_score",
]


def fetch_all_rows(table_name):
    print("\n" + "=" * 70)
    print(f"FETCHING: {table_name}")
    print("=" * 70)

    rows_all = []
    offset = 0

    while True:
        try:
            response = (
                supabase.table(table_name)
                .select("*")
                .range(offset, offset + PAGE_SIZE - 1)
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
        print(f"Retrieved {len(rows_all)} rows")

        if len(rows) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

    print(f"Total rows retrieved: {len(rows_all)}")
    return pd.DataFrame(rows_all)


def prepare_features(df):
    print("\n" + "=" * 70)
    print("PREPARING CLUSTERING FEATURES")
    print("=" * 70)

    missing_features = [c for c in FEATURES if c not in df.columns]

    if missing_features:
        print("ERROR: Missing required features:")
        for col in missing_features:
            print(f"  - {col}")
        sys.exit(1)

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("\nMissing values before imputation:")
    total_missing = 0

    for col in FEATURES:
        count = int(df[col].isna().sum())
        if count > 0:
            print(f"{col}: {count}")
            total_missing += count

    X = df[FEATURES].copy()

    # Same median-imputation logic as Step 8C.
    imputation_values = {}

    for col in FEATURES:
        # Treat infinities as missing before calculating the median.
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)

        median_value = X[col].median()

        if pd.isna(median_value):
            print(f"ERROR: Feature {col} contains no usable values.")
            sys.exit(1)

        imputation_values[col] = float(median_value)
        X[col] = X[col].fillna(median_value)

    if X.isna().sum().sum() > 0 or np.isinf(X.to_numpy()).any():
        print("ERROR: Invalid values remain after preprocessing.")
        sys.exit(1)

    print(f"Rows used for evaluation: {len(X)}")
    print(f"Features used: {len(FEATURES)}")
    print(f"Total missing values handled: {total_missing}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("StandardScaler applied successfully.")

    return X, X_scaled, imputation_values


def calculate_metrics(X_scaled, labels):
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return {
            "silhouette_score": np.nan,
            "davies_bouldin_index": np.nan,
            "calinski_harabasz_index": np.nan,
            "cluster_count": len(unique_labels),
        }

    return {
        "silhouette_score": float(
            silhouette_score(X_scaled, labels)
        ),
        "davies_bouldin_index": float(
            davies_bouldin_score(X_scaled, labels)
        ),
        "calinski_harabasz_index": float(
            calinski_harabasz_score(X_scaled, labels)
        ),
        "cluster_count": len(unique_labels),
    }


def print_and_return_result(algorithm, k, X_scaled, labels):
    metrics = calculate_metrics(X_scaled, labels)

    counts = pd.Series(labels).value_counts().sort_index()

    print(f"Silhouette Score       : {metrics['silhouette_score']:.4f}")
    print(f"Davies-Bouldin Index   : {metrics['davies_bouldin_index']:.4f}")
    print(
        f"Calinski-Harabasz Index: "
        f"{metrics['calinski_harabasz_index']:.2f}"
    )

    print("\nCluster sizes:")
    for cluster_id, count in counts.items():
        print(f"  Cluster {cluster_id}: {count}")

    return {
        "algorithm": algorithm,
        "k": k,
        **metrics,
        "cluster_sizes": ",".join(str(int(x)) for x in counts.values),
    }


def evaluate_kmeans(X_scaled, k):
    print("\n" + "-" * 70)
    print(f"K-MEANS — K={k}")
    print("-" * 70)

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )
    labels = model.fit_predict(X_scaled)

    return print_and_return_result(
        "K-Means", k, X_scaled, labels
    )


def evaluate_agglomerative(X_scaled, k):
    print("\n" + "-" * 70)
    print(f"AGGLOMERATIVE / HIERARCHICAL — K={k}")
    print("-" * 70)

    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )
    labels = model.fit_predict(X_scaled)

    return print_and_return_result(
        "Agglomerative", k, X_scaled, labels
    )


def evaluate_gaussian_mixture(X_scaled, k):
    print("\n" + "-" * 70)
    print(f"GAUSSIAN MIXTURE MODEL — K={k}")
    print("-" * 70)

    model = GaussianMixture(
        n_components=k,
        random_state=42,
        n_init=10
    )
    model.fit(X_scaled)
    labels = model.predict(X_scaled)

    return print_and_return_result(
        "Gaussian Mixture", k, X_scaled, labels
    )


def rank_models(results_df):
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df[
            [
                "algorithm",
                "k",
                "silhouette_score",
                "davies_bouldin_index",
                "calinski_harabasz_index",
                "cluster_sizes",
            ]
        ].to_string(index=False)
    )

    # Higher silhouette = better.
    results_df["silhouette_rank"] = results_df[
        "silhouette_score"
    ].rank(ascending=False, method="min")

    # Lower Davies-Bouldin = better.
    results_df["davies_bouldin_rank"] = results_df[
        "davies_bouldin_index"
    ].rank(ascending=True, method="min")

    # Higher Calinski-Harabasz = better.
    results_df["calinski_harabasz_rank"] = results_df[
        "calinski_harabasz_index"
    ].rank(ascending=False, method="min")

    # Lower average rank = better overall.
    results_df["average_metric_rank"] = results_df[
        [
            "silhouette_rank",
            "davies_bouldin_rank",
            "calinski_harabasz_rank",
        ]
    ].mean(axis=1)

    ranked = results_df.sort_values(
        ["average_metric_rank", "silhouette_score"],
        ascending=[True, False]
    ).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("RANKED RESULTS")
    print("=" * 70)

    print(
        ranked[
            [
                "algorithm",
                "k",
                "silhouette_score",
                "davies_bouldin_index",
                "calinski_harabasz_index",
                "average_metric_rank",
            ]
        ].to_string(index=False)
    )

    best = ranked.iloc[0]

    print("\n" + "=" * 70)
    print("BEST OVERALL CONFIGURATION BY AVERAGE METRIC RANK")
    print("=" * 70)

    print(f"Algorithm: {best['algorithm']}")
    print(f"K: {int(best['k'])}")
    print(f"Silhouette Score: {best['silhouette_score']:.4f}")
    print(
        f"Davies-Bouldin Index: "
        f"{best['davies_bouldin_index']:.4f}"
    )
    print(
        f"Calinski-Harabasz Index: "
        f"{best['calinski_harabasz_index']:.2f}"
    )
    print(f"Average Metric Rank: {best['average_metric_rank']:.2f}")

    print("\nNOTE: Metric ranking is only a screening step.")
    print(
        "Also check cluster sizes and business interpretability "
        "before replacing the current Step 8C model."
    )

    return ranked


def save_results(results_df, imputation_values):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    info = {
        "source_table": SOURCE_TABLE,
        "features": FEATURES,
        "preprocessing": [
            "numeric conversion",
            "replace infinite values with NaN",
            "median imputation",
            "StandardScaler",
        ],
        "imputation_medians": imputation_values,
        "kmeans_k_values": KMEANS_K_VALUES,
        "comparison_k": COMPARISON_K,
        "random_state": 42,
    }

    info_path = os.path.join(
        OUTPUT_DIR,
        "clustering_evaluation_info.json"
    )

    with open(info_path, "w", encoding="utf-8") as file:
        json.dump(info, file, indent=2)

    print("\n" + "=" * 70)
    print("SAVING EVALUATION RESULTS")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_CSV}")
    print(f"Preprocessing information saved to: {info_path}")


def main():
    print("\n" + "=" * 70)
    print("CLUSTERING MODEL EVALUATION")
    print("=" * 70)
    print(
        "Exact Step 8C features + preprocessing; "
        "read-only evaluation."
    )

    df = fetch_all_rows(SOURCE_TABLE)

    if df.empty:
        print("ERROR: Source table is empty.")
        sys.exit(1)

    if {"ACO_ID", "performance_year"}.issubset(df.columns):
        duplicates = df.duplicated(
            subset=["ACO_ID", "performance_year"]
        ).sum()

        print(f"Duplicate ACO-Year rows: {duplicates}")

        if duplicates > 0:
            print("ERROR: Duplicate ACO-Year rows found.")
            sys.exit(1)

    _, X_scaled, imputation_values = prepare_features(df)

    results = []

    # K-Means K=2, K=3, K=4.
    for k in KMEANS_K_VALUES:
        results.append(
            evaluate_kmeans(X_scaled, k)
        )

    # Alternative algorithms at K=3.
    results.append(
        evaluate_agglomerative(
            X_scaled,
            COMPARISON_K
        )
    )

    results.append(
        evaluate_gaussian_mixture(
            X_scaled,
            COMPARISON_K
        )
    )

    results_df = pd.DataFrame(results)

    ranked_df = rank_models(results_df)

    save_results(
        ranked_df,
        imputation_values
    )

    print("\n" + "=" * 70)
    print("CLUSTERING EVALUATION COMPLETE")
    print("=" * 70)
    print("No source data was modified.")
    print("No Supabase result tables were modified.")
    print("No segmentation results were inserted.")
    print(f"Evaluation file: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()