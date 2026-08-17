import os
from dotenv import load_dotenv
from supabase import create_client
import pandas as pd


# --------------------------------------------------
# 1. Load Supabase credentials
# --------------------------------------------------

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL or SUPABASE_KEY is missing.")

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# --------------------------------------------------
# 2. Read a table
# --------------------------------------------------

def read_table(table_name, columns, page_size=1000):
    
    all_rows = []

    start = 0

    while True:

        end = start + page_size - 1

        response = (
            supabase
            .table(table_name)
            .select(columns)
            .range(start, end)
            .execute()
        )

        rows = response.data

        if not rows:
            break

        all_rows.extend(rows)

        print(
            f"{table_name}: retrieved "
            f"{len(all_rows)} rows"
        )

        if len(rows) < page_size:
            break

        start += page_size

    return pd.DataFrame(all_rows)


# --------------------------------------------------
# 3. Check ACO + Year uniqueness
# --------------------------------------------------

def check_aco_year(
    df,
    table_name,
    aco_column,
    year_column
):

    print("\n" + "=" * 60)
    print(table_name)
    print("=" * 60)

    print("Rows:", len(df))

    print(
        "Unique ACOs:",
        df[aco_column].nunique()
    )

    print(
        "Years:",
        sorted(
            df[year_column]
            .dropna()
            .unique()
            .tolist()
        )
    )

    duplicates = df[
        df.duplicated(
            subset=[aco_column, year_column],
            keep=False
        )
    ]

    print(
        "Duplicate ACO-Year rows:",
        len(duplicates)
    )

    if len(duplicates) > 0:

        print("\nDuplicate examples:")

        print(
            duplicates[
                [aco_column, year_column]
            ]
            .sort_values(
                [aco_column, year_column]
            )
            .head(20)
            .to_string(index=False)
        )

    else:

        print(
            "STATUS: ACO + Year is unique."
        )


# --------------------------------------------------
# 4. fact_aco_performance
# --------------------------------------------------

fact = read_table(
    "fact_aco_performance",
    "ACO_ID,performance_year"
)

check_aco_year(
    fact,
    "fact_aco_performance",
    "ACO_ID",
    "performance_year"
)


# --------------------------------------------------
# 5. Financial feature table
# --------------------------------------------------

financial = read_table(
    "aco_financial_ml_training",
    "ACO_ID,feature_year,target_year"
)

check_aco_year(
    financial,
    "aco_financial_ml_training",
    "ACO_ID",
    "target_year"
)


# --------------------------------------------------
# 6. Quality / analytics table
# --------------------------------------------------

quality = read_table(
    "aco_analytics",
    "aco_id,performance_year"
)

check_aco_year(
    quality,
    "aco_analytics",
    "aco_id",
    "performance_year"
)