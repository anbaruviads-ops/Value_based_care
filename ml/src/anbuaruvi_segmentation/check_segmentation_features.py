import os
import sys
import time
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

TABLES = [
    "aco_utilization_ml_features",
    "aco_financial_ml_training",
    "aco_analytics"
]

PAGE_SIZE = 1000


# ============================================================
# FETCH COLUMNS
# ============================================================

def fetch_columns(table_name):

    print()
    print("=" * 70)
    print(f"TABLE: {table_name}")
    print("=" * 70)

    rows = []
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

            print(f"ERROR reading {table_name}")
            print(e)
            sys.exit(1)

        data = response.data

        if not data:
            break

        rows.extend(data)

        print(
            f"Retrieved {len(rows)} rows"
        )

        if len(data) < PAGE_SIZE:
            break

        offset += PAGE_SIZE
        time.sleep(0.1)

    if not rows:

        print("WARNING: Table contains no rows.")
        return

    columns = list(rows[0].keys())

    print()
    print(f"Rows: {len(rows)}")
    print(f"Columns: {len(columns)}")
    print()

    for i, column in enumerate(
        columns,
        start=1
    ):

        print(
            f"{i:03d}. {column}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("STEP 7D — SEGMENTATION FEATURE AVAILABILITY CHECK")
    print("=" * 70)

    for table in TABLES:

        fetch_columns(table)

    print()
    print("=" * 70)
    print("STEP 7D CHECK COMPLETE")
    print("=" * 70)

    print()
    print(
        "No data was modified."
    )

    print(
        "No rows were inserted."
    )


if __name__ == "__main__":
    main()