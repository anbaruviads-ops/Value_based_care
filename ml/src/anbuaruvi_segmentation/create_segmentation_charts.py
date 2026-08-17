import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from supabase import create_client, Client

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("ERROR: SUPABASE_URL or SUPABASE_KEY is missing.")
    raise SystemExit(1)

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

SOURCE_TABLE = "aco_segmentation_ml_features"
RESULT_TABLE = "aco_segmentation_results"

EVALUATION_FILE = "evaluation/clustering_model_comparison.csv"

MODEL_FILE = "models/aco_segmentation_kmeans.pkl"

OUTPUT_DIR = "evaluation/charts"

PAGE_SIZE = 1000


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
    "average_available_risk_score",
]


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FETCH DATA
# ============================================================

def fetch_all_rows(table_name):

    print()
    print("=" * 70)
    print(f"FETCHING: {table_name}")
    print("=" * 70)

    rows_all = []
    offset = 0

    while True:

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
# LOAD EVALUATION RESULTS
# ============================================================

def load_evaluation_results():

    print()
    print("=" * 70)
    print("LOADING MODEL EVALUATION RESULTS")
    print("=" * 70)

    if not os.path.exists(EVALUATION_FILE):

        print(
            f"ERROR: File not found: {EVALUATION_FILE}"
        )

        raise SystemExit(1)

    evaluation = pd.read_csv(
        EVALUATION_FILE
    )

    print()
    print(evaluation)

    return evaluation


# ============================================================
# CHART 1
# SILHOUETTE SCORE COMPARISON
# ============================================================

def chart_silhouette(evaluation):

    print()
    print("Creating Chart 1 — Silhouette Score")

    data = evaluation.copy()

    labels = (
        data["algorithm"]
        + " K="
        + data["k"].astype(str)
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        labels,
        data["silhouette_score"]
    )

    plt.ylabel(
        "Silhouette Score"
    )

    plt.xlabel(
        "Clustering Configuration"
    )

    plt.title(
        "Clustering Model Comparison — Silhouette Score"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "01_silhouette_score_comparison.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# CHART 2
# DAVIES-BOULDIN INDEX
# ============================================================

def chart_davies_bouldin(evaluation):

    print()
    print("Creating Chart 2 — Davies-Bouldin Index")

    data = evaluation.copy()

    labels = (
        data["algorithm"]
        + " K="
        + data["k"].astype(str)
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        labels,
        data["davies_bouldin_index"]
    )

    plt.ylabel(
        "Davies-Bouldin Index"
    )

    plt.xlabel(
        "Clustering Configuration"
    )

    plt.title(
        "Clustering Model Comparison — Davies-Bouldin Index"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "02_davies_bouldin_comparison.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# CHART 3
# CALINSKI-HARABASZ INDEX
# ============================================================

def chart_calinski(evaluation):

    print()
    print("Creating Chart 3 — Calinski-Harabasz Index")

    data = evaluation.copy()

    labels = (
        data["algorithm"]
        + " K="
        + data["k"].astype(str)
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        labels,
        data["calinski_harabasz_index"]
    )

    plt.ylabel(
        "Calinski-Harabasz Index"
    )

    plt.xlabel(
        "Clustering Configuration"
    )

    plt.title(
        "Clustering Model Comparison — Calinski-Harabasz Index"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "03_calinski_harabasz_comparison.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# CHART 4
# K-MEANS K COMPARISON
# ============================================================

def chart_kmeans_k_comparison(evaluation):

    print()
    print("Creating Chart 4 — K-Means K Comparison")

    data = evaluation[
        evaluation["algorithm"] == "K-Means"
    ].copy()

    data = data.sort_values(
        "k"
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        data["k"],
        data["silhouette_score"],
        marker="o",
        label="Silhouette Score"
    )

    plt.plot(
        data["k"],
        data["davies_bouldin_index"],
        marker="o",
        label="Davies-Bouldin Index"
    )

    plt.xlabel(
        "Number of Clusters (K)"
    )

    plt.ylabel(
        "Metric Value"
    )

    plt.title(
        "K-Means Performance Across Different K Values"
    )

    plt.xticks(
        data["k"]
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "04_kmeans_k_comparison.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# PREPARE FINAL CLUSTER DATA
# ============================================================

def prepare_cluster_data(
    source_df,
    result_df
):

    print()
    print("=" * 70)
    print("PREPARING FINAL CLUSTER DATA")
    print("=" * 70)

    merged = source_df[
        [
            "ACO_ID",
            "performance_year"
        ] + FEATURES
    ].merge(
        result_df[
            [
                "ACO_ID",
                "performance_year",
                "cluster_id",
                "performance_segment",
                "performance_index"
            ]
        ],
        on=[
            "ACO_ID",
            "performance_year"
        ],
        how="inner"
    )

    print(
        f"Merged rows: {len(merged)}"
    )

    print()
    print(
        "Cluster distribution:"
    )

    print(
        merged["performance_segment"]
        .value_counts()
    )

    return merged


# ============================================================
# CHART 5
# FINAL SEGMENT DISTRIBUTION
# ============================================================

def chart_segment_distribution(cluster_df):

    print()
    print("Creating Chart 5 — Segment Distribution")

    counts = (
        cluster_df[
            "performance_segment"
        ]
        .value_counts()
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.bar(
        counts.index,
        counts.values
    )

    plt.ylabel(
        "Number of ACO-Year Records"
    )

    plt.xlabel(
        "Performance Segment"
    )

    plt.title(
        "ACO Performance Segmentation Distribution"
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "05_final_segment_distribution.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# CHART 6
# CLUSTER PERFORMANCE PROFILE
# ============================================================

def chart_cluster_profile(cluster_df):

    print()
    print("Creating Chart 6 — Cluster Performance Profile")

    metrics = [
        "SavingsLossPct",
        "ExpenditureVariancePct",
        "quality_score",
        "utilization_score",
        "average_available_risk_score"
    ]

    profile = (
        cluster_df
        .groupby(
            "performance_segment"
        )[metrics]
        .mean()
    )

    # Normalize each metric to 0–1
    normalized = profile.copy()

    for column in metrics:

        minimum = profile[column].min()
        maximum = profile[column].max()

        if maximum != minimum:

            normalized[column] = (
                profile[column] - minimum
            ) / (
                maximum - minimum
            )

        else:

            normalized[column] = 0.5

    plt.figure(
        figsize=(11, 6)
    )

    for segment in normalized.index:

        plt.plot(
            metrics,
            normalized.loc[segment],
            marker="o",
            label=segment
        )

    plt.ylabel(
        "Normalized Mean Value"
    )

    plt.xlabel(
        "Performance Dimensions"
    )

    plt.title(
        "ACO Segment Performance Profile"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "06_cluster_performance_profile.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# CHART 7
# PCA VISUALIZATION
# ============================================================

def chart_pca(cluster_df):

    print()
    print("Creating Chart 7 — PCA Cluster Visualization")

    X = cluster_df[
        FEATURES
    ].copy()

    # Convert numeric
    for column in FEATURES:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # Replace infinite values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Median imputation
    X = X.fillna(
        X.median()
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    pca = PCA(
        n_components=2,
        random_state=42
    )

    X_pca = pca.fit_transform(
        X_scaled
    )

    plot_df = pd.DataFrame(
        {
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "segment":
                cluster_df[
                    "performance_segment"
                ].values
        }
    )

    plt.figure(
        figsize=(10, 7)
    )

    for segment in sorted(
        plot_df["segment"].unique()
    ):

        subset = plot_df[
            plot_df["segment"] == segment
        ]

        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            label=segment,
            alpha=0.65
        )

    explained_1 = (
        pca.explained_variance_ratio_[0]
        * 100
    )

    explained_2 = (
        pca.explained_variance_ratio_[1]
        * 100
    )

    plt.xlabel(
        f"PC1 ({explained_1:.1f}% variance)"
    )

    plt.ylabel(
        f"PC2 ({explained_2:.1f}% variance)"
    )

    plt.title(
        "ACO Performance Segmentation — PCA Visualization"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "07_pca_cluster_visualization.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# CHART 8
# SAVINGS VS QUALITY
# ============================================================

def chart_savings_quality(cluster_df):

    print()
    print("Creating Chart 8 — Savings vs Quality")

    plt.figure(
        figsize=(10, 7)
    )

    for segment in sorted(
        cluster_df[
            "performance_segment"
        ].unique()
    ):

        subset = cluster_df[
            cluster_df[
                "performance_segment"
            ] == segment
        ]

        plt.scatter(
            subset["SavingsLossPct"],
            subset["quality_score"],
            label=segment,
            alpha=0.65
        )

    plt.xlabel(
        "Savings / Loss (%)"
    )

    plt.ylabel(
        "Quality Score"
    )

    plt.title(
        "ACO Financial Performance vs Quality"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "08_savings_vs_quality.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# CHART 9
# UTILIZATION VS QUALITY
# ============================================================

def chart_utilization_quality(cluster_df):

    print()
    print(
        "Creating Chart 9 — Utilization vs Quality"
    )

    plt.figure(
        figsize=(10, 7)
    )

    for segment in sorted(
        cluster_df[
            "performance_segment"
        ].unique()
    ):

        subset = cluster_df[
            cluster_df[
                "performance_segment"
            ] == segment
        ]

        plt.scatter(
            subset["utilization_score"],
            subset["quality_score"],
            label=segment,
            alpha=0.65
        )

    plt.xlabel(
        "Utilization Score"
    )

    plt.ylabel(
        "Quality Score"
    )

    plt.title(
        "ACO Utilization vs Quality"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "09_utilization_vs_quality.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# CHART 10
# RISK VS SAVINGS
# ============================================================

def chart_risk_savings(cluster_df):

    print()
    print(
        "Creating Chart 10 — Risk vs Savings"
    )

    plt.figure(
        figsize=(10, 7)
    )

    for segment in sorted(
        cluster_df[
            "performance_segment"
        ].unique()
    ):

        subset = cluster_df[
            cluster_df[
                "performance_segment"
            ] == segment
        ]

        plt.scatter(
            subset[
                "average_available_risk_score"
            ],
            subset["SavingsLossPct"],
            label=segment,
            alpha=0.65
        )

    plt.xlabel(
        "Average Available Risk Score"
    )

    plt.ylabel(
        "Savings / Loss (%)"
    )

    plt.title(
        "ACO Risk Profile vs Financial Performance"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "10_risk_vs_savings.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# GENERATE SUMMARY
# ============================================================

def create_summary(
    evaluation,
    cluster_df
):

    print()
    print(
        "Creating segmentation summary..."
    )

    summary = (
        cluster_df
        .groupby(
            "performance_segment"
        )
        .agg(
            ACO_Year_Count=(
                "ACO_ID",
                "count"
            ),
            Unique_ACOs=(
                "ACO_ID",
                "nunique"
            ),
            Avg_SavingsLossPct=(
                "SavingsLossPct",
                "mean"
            ),
            Avg_ExpenditureVariancePct=(
                "ExpenditureVariancePct",
                "mean"
            ),
            Avg_QualityScore=(
                "quality_score",
                "mean"
            ),
            Avg_UtilizationScore=(
                "utilization_score",
                "mean"
            ),
            Avg_RiskScore=(
                "average_available_risk_score",
                "mean"
            ),
            Avg_PerformanceIndex=(
                "performance_index",
                "mean"
            )
        )
        .reset_index()
    )

    path = os.path.join(
        OUTPUT_DIR,
        "segmentation_business_summary.csv"
    )

    summary.to_csv(
        path,
        index=False
    )

    print()
    print(summary.round(3))

    print()
    print(
        f"Summary saved: {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "ACO SEGMENTATION VISUALIZATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    evaluation = (
        load_evaluation_results()
    )

    # --------------------------------------------------------
    # Model comparison charts
    # --------------------------------------------------------

    chart_silhouette(
        evaluation
    )

    chart_davies_bouldin(
        evaluation
    )

    chart_calinski(
        evaluation
    )

    chart_kmeans_k_comparison(
        evaluation
    )

    # --------------------------------------------------------
    # Fetch source and results
    # --------------------------------------------------------

    source_df = fetch_all_rows(
        SOURCE_TABLE
    )

    result_df = fetch_all_rows(
        RESULT_TABLE
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    cluster_df = prepare_cluster_data(
        source_df,
        result_df
    )

    # --------------------------------------------------------
    # Final segmentation charts
    # --------------------------------------------------------

    chart_segment_distribution(
        cluster_df
    )

    chart_cluster_profile(
        cluster_df
    )

    chart_pca(
        cluster_df
    )

    chart_savings_quality(
        cluster_df
    )

    chart_utilization_quality(
        cluster_df
    )

    chart_risk_savings(
        cluster_df
    )

    # --------------------------------------------------------
    # Business summary
    # --------------------------------------------------------

    create_summary(
        evaluation,
        cluster_df
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "VISUALIZATION COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        f"All charts saved in: {OUTPUT_DIR}"
    )

    print()
    print(
        "No training performed."
    )

    print(
        "No source data modified."
    )

    print(
        "No Supabase result data modified."
    )


if __name__ == "__main__":
    main()