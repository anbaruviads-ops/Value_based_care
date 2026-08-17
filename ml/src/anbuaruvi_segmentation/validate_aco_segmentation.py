import os
import sys
import time
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

TABLE_NAME = "aco_segmentation_results"
PAGE_SIZE = 1000


# ============================================================
# FETCH ALL RESULTS
# ============================================================

def fetch_all_rows():

    print()
    print("=" * 70)
    print(f"FETCHING: {TABLE_NAME}")
    print("=" * 70)

    columns = [
        "id",
        "ACO_ID",
        "performance_year",
        "cluster_id",
        "performance_segment",
        "performance_index",
        "SavingsLossPct",
        "ExpenditureVariancePct",
        "quality_score",
        "utilization_score",
        "created_at",
    ]

    all_rows = []
    offset = 0

    while True:

        try:

            response = (
                supabase
                .table(TABLE_NAME)
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
            f"Retrieved {len(all_rows)} rows"
        )

        if len(rows) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

        time.sleep(0.1)

    return pd.DataFrame(all_rows)


# ============================================================
# VALIDATION
# ============================================================

def validate_results(df):

    print()
    print("=" * 70)
    print("STEP 8D — SEGMENTATION RESULT VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BASIC VALIDATION")
    print("=" * 70)

    print(f"Rows: {len(df)}")

    print(
        f"Unique ACOs: "
        f"{df['ACO_ID'].nunique()}"
    )

    print(
        f"Years: "
        f"{sorted(df['performance_year'].dropna().unique())}"
    )

    # --------------------------------------------------------
    # DUPLICATE CHECK
    # --------------------------------------------------------

    duplicate_count = df.duplicated(
        subset=[
            "ACO_ID",
            "performance_year"
        ]
    ).sum()

    print(
        f"Duplicate ACO-Year rows: "
        f"{duplicate_count}"
    )

    if duplicate_count == 0:
        print("STATUS: PASS — ACO + Year is unique.")
    else:
        print("STATUS: FAIL — Duplicate ACO-Year rows found.")

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MISSING VALUE CHECK")
    print("=" * 70)

    required_columns = [
        "ACO_ID",
        "performance_year",
        "cluster_id",
        "performance_segment",
        "performance_index",
        "SavingsLossPct",
        "ExpenditureVariancePct",
        "quality_score",
        "utilization_score",
    ]

    missing_found = False

    for col in required_columns:

        count = df[col].isna().sum()

        print(
            f"{col:35s}: {count}"
        )

        if count > 0:
            missing_found = True

    if not missing_found:
        print()
        print("STATUS: PASS — No missing required values.")

    # --------------------------------------------------------
    # CLUSTER CHECK
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CLUSTER VALIDATION")
    print("=" * 70)

    print(
        df["cluster_id"]
        .value_counts()
        .sort_index()
    )

    expected_clusters = {0, 1, 2}

    actual_clusters = set(
        df["cluster_id"]
        .dropna()
        .astype(int)
        .unique()
    )

    if actual_clusters == expected_clusters:
        print(
            "STATUS: PASS — Clusters 0, 1 and 2 are present."
        )
    else:
        print(
            "WARNING: Unexpected cluster IDs:",
            actual_clusters
        )

    # --------------------------------------------------------
    # BUSINESS SEGMENT CHECK
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BUSINESS SEGMENT DISTRIBUTION")
    print("=" * 70)

    print(
        df["performance_segment"]
        .value_counts(dropna=False)
    )

    expected_segments = {
        "High Performing",
        "Moderate",
        "Needs Attention"
    }

    actual_segments = set(
        df["performance_segment"]
        .dropna()
        .unique()
    )

    if actual_segments == expected_segments:
        print(
            "STATUS: PASS — All expected business segments exist."
        )
    else:
        print(
            "WARNING: Unexpected business segments:",
            actual_segments
        )

    # --------------------------------------------------------
    # CLUSTER → SEGMENT MAPPING
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CLUSTER → BUSINESS SEGMENT MAPPING")
    print("=" * 70)

    mapping = (
        df[
            [
                "cluster_id",
                "performance_segment"
            ]
        ]
        .drop_duplicates()
        .sort_values("cluster_id")
    )

    print(mapping.to_string(index=False))

    # --------------------------------------------------------
    # PERFORMANCE INDEX
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PERFORMANCE INDEX")
    print("=" * 70)

    print(
        df.groupby(
            "performance_segment"
        )["performance_index"]
        .agg([
            "count",
            "mean",
            "min",
            "max"
        ])
        .round(4)
    )

    # --------------------------------------------------------
    # CLUSTER PROFILES
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CLUSTER PROFILES")
    print("=" * 70)

    profile_columns = [
        "SavingsLossPct",
        "ExpenditureVariancePct",
        "quality_score",
        "utilization_score",
        "performance_index",
    ]

    cluster_profiles = (
        df
        .groupby("cluster_id")[profile_columns]
        .mean()
        .round(4)
    )

    print(cluster_profiles)

    # --------------------------------------------------------
    # BUSINESS SEGMENT PROFILES
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BUSINESS SEGMENT PROFILES")
    print("=" * 70)

    segment_profiles = (
        df
        .groupby("performance_segment")[profile_columns]
        .mean()
        .round(4)
    )

    print(segment_profiles)

    # --------------------------------------------------------
    # NEEDS ATTENTION CHECK
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("NEEDS ATTENTION VALIDATION")
    print("=" * 70)

    attention = df[
        df["performance_segment"]
        == "Needs Attention"
    ]

    print(
        f"Needs Attention rows: "
        f"{len(attention)}"
    )

    print(
        f"Unique ACOs: "
        f"{attention['ACO_ID'].nunique()}"
    )

    if len(attention) > 0:

        print()
        print(
            attention[
                [
                    "ACO_ID",
                    "performance_year",
                    "performance_index",
                    "SavingsLossPct",
                    "ExpenditureVariancePct",
                    "quality_score",
                    "utilization_score",
                ]
            ]
            .sort_values(
                "performance_index"
            )
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # SCORE RANGE CHECK
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SCORE RANGE VALIDATION")
    print("=" * 70)

    score_columns = [
        "performance_index",
        "quality_score",
        "utilization_score",
    ]

    for col in score_columns:

        minimum = df[col].min()
        maximum = df[col].max()

        print(
            f"{col:30s}"
            f"min={minimum:.4f} "
            f"max={maximum:.4f}"
        )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 8D VALIDATION COMPLETE")
    print("=" * 70)

    if (
        duplicate_count == 0
        and not missing_found
        and actual_clusters == expected_clusters
        and actual_segments == expected_segments
    ):

        print()
        print("OVERALL STATUS: PASS")

    else:

        print()
        print("OVERALL STATUS: REVIEW REQUIRED")

    print()
    print("No data was modified.")
    print("No rows were inserted.")
    print("No rows were deleted.")


# ============================================================
# MAIN
# ============================================================

def main():

    df = fetch_all_rows()

    if df.empty:

        print()
        print("ERROR: No segmentation results found.")

        sys.exit(1)

    validate_results(df)


if __name__ == "__main__":
    main()