import pandas as pd

# We will replace these with Supabase retrieval later.
# For now this script is only for checking the key structure.

def check_key(df, aco_col, year_col, table_name):
    print(f"\n===== {table_name} =====")

    print("Rows:", len(df))
    print("Unique ACOs:", df[aco_col].nunique())
    print("Years:", sorted(df[year_col].dropna().unique()))

    duplicate_count = df.duplicated(
        subset=[aco_col, year_col]
    ).sum()

    print("Duplicate ACO-Year rows:", duplicate_count)

    if duplicate_count > 0:
        print("\nDuplicate examples:")
        print(
            df[
                df.duplicated(
                    subset=[aco_col, year_col],
                    keep=False
                )
            ][[aco_col, year_col]]
            .sort_values([aco_col, year_col])
            .head(20)
        )