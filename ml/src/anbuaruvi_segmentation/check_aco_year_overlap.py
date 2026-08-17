import os
from dotenv import load_dotenv
from supabase import create_client
import pandas as pd


# --------------------------------------------------
# 1. Supabase connection
# --------------------------------------------------

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials are missing.")

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# --------------------------------------------------
# 2. Read complete table using pagination
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

        if len(rows) < page_size:
            break

        start += page_size

    return pd.DataFrame(all_rows)


# --------------------------------------------------
# 3. Load only ACO + Year
# --------------------------------------------------

fact = read_table(
    "fact_aco_performance",
    "ACO_ID,performance_year"
)

financial = read_table(
    "aco_financial_ml_training",
    "ACO_ID,target_year"
)

quality = read_table(
    "aco_analytics",
    "aco_id,performance_year"
)


# --------------------------------------------------
# 4. Standardize column names
# --------------------------------------------------

fact = fact.rename(
    columns={
        "ACO_ID": "ACO_ID",
        "performance_year": "YEAR"
    }
)

financial = financial.rename(
    columns={
        "ACO_ID": "ACO_ID",
        "target_year": "YEAR"
    }
)

quality = quality.rename(
    columns={
        "aco_id": "ACO_ID",
        "performance_year": "YEAR"
    }
)


# --------------------------------------------------
# 5. Make sure data types match
# --------------------------------------------------

fact["ACO_ID"] = fact["ACO_ID"].astype(str)
financial["ACO_ID"] = financial["ACO_ID"].astype(str)
quality["ACO_ID"] = quality["ACO_ID"].astype(str)

fact["YEAR"] = pd.to_numeric(
    fact["YEAR"],
    errors="coerce"
)

financial["YEAR"] = pd.to_numeric(
    financial["YEAR"],
    errors="coerce"
)

quality["YEAR"] = pd.to_numeric(
    quality["YEAR"],
    errors="coerce"
)


# --------------------------------------------------
# 6. Create ACO-Year keys
# --------------------------------------------------

fact_keys = set(
    zip(
        fact["ACO_ID"],
        fact["YEAR"]
    )
)

financial_keys = set(
    zip(
        financial["ACO_ID"],
        financial["YEAR"]
    )
)

quality_keys = set(
    zip(
        quality["ACO_ID"],
        quality["YEAR"]
    )
)


# --------------------------------------------------
# 7. Calculate overlaps
# --------------------------------------------------

all_three = (
    fact_keys
    & financial_keys
    & quality_keys
)

fact_financial = (
    fact_keys
    & financial_keys
)

fact_quality = (
    fact_keys
    & quality_keys
)

financial_quality = (
    financial_keys
    & quality_keys
)


# --------------------------------------------------
# 8. Print results
# --------------------------------------------------

print("\n" + "=" * 70)
print("ACO-YEAR OVERLAP ANALYSIS")
print("=" * 70)

print(
    f"\nfact_aco_performance: {len(fact_keys)}"
)

print(
    f"aco_financial_ml_training: {len(financial_keys)}"
)

print(
    f"aco_analytics: {len(quality_keys)}"
)

print("\n----------------------------------------")

print(
    f"Fact ∩ Financial: {len(fact_financial)}"
)

print(
    f"Fact ∩ Quality: {len(fact_quality)}"
)

print(
    f"Financial ∩ Quality: {len(financial_quality)}"
)

print(
    f"\nALL THREE TABLES: {len(all_three)}"
)


# --------------------------------------------------
# 9. Year-wise overlap
# --------------------------------------------------

print("\n" + "=" * 70)
print("YEAR-WISE OVERLAP")
print("=" * 70)

years = sorted(
    set(fact["YEAR"].dropna())
    | set(financial["YEAR"].dropna())
    | set(quality["YEAR"].dropna())
)

for year in years:

    fact_year = {
        aco
        for aco, y in fact_keys
        if y == year
    }

    financial_year = {
        aco
        for aco, y in financial_keys
        if y == year
    }

    quality_year = {
        aco
        for aco, y in quality_keys
        if y == year
    }

    common_year = (
        fact_year
        & financial_year
        & quality_year
    )

    print(
        f"\n{int(year)}"
    )

    print(
        f"  Fact:       {len(fact_year)}"
    )

    print(
        f"  Financial:  {len(financial_year)}"
    )

    print(
        f"  Quality:    {len(quality_year)}"
    )

    print(
        f"  All three:  {len(common_year)}"
    )


# --------------------------------------------------
# 10. Show examples of missing financial years
# --------------------------------------------------

missing_financial = (
    fact_keys - financial_keys
)

print("\n" + "=" * 70)
print("EXAMPLES: FACT ACO-YEARS NOT IN FINANCIAL TABLE")
print("=" * 70)

for item in sorted(
    missing_financial,
    key=lambda x: (x[1], x[0])
)[:20]:

    print(item)


# --------------------------------------------------
# 11. Show examples missing from quality table
# --------------------------------------------------

missing_quality = (
    fact_keys - quality_keys
)

print("\n" + "=" * 70)
print("EXAMPLES: FACT ACO-YEARS NOT IN QUALITY TABLE")
print("=" * 70)

for item in sorted(
    missing_quality,
    key=lambda x: (x[1], x[0])
)[:20]:

    print(item)