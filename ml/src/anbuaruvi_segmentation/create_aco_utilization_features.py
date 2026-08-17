import os
import sys
import time
import numpy as np
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
    print("Check your .env file.")
    sys.exit(1)


supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


SOURCE_TABLE = "fact_aco_performance"
TARGET_TABLE = "aco_utilization_ml_features"

PAGE_SIZE = 1000
INSERT_BATCH_SIZE = 500


# ============================================================
# SOURCE FEATURES
# ============================================================

SOURCE_FEATURES = [

    "ACO_ID",
    "performance_year",

    "N_AB",

    "P_EDV_Vis",
    "P_EDV_Vis_HOSP",

    "ADM",
    "ADM_S_Trm",
    "ADM_L_Trm",
    "ADM_Rehab",
    "ADM_Psych",

    "P_SNF_ADM",
    "SNF_LOS",
    "SNF_PayperStay",

    "chf_adm",
    "copd_adm",

    "prov_Rate_1000",

    "P_CT_VIS",
    "P_MRI_VIS",

    "P_EM_Total",
    "P_EM_PCP_Vis",
    "P_EM_SP_Vis",

    "P_Nurse_Vis",
    "P_FQHC_RHC_Vis",
]


# ============================================================
# FINAL FEATURE COLUMNS
# EXACTLY 62 COLUMNS
# ============================================================

FINAL_COLUMNS = [

    # 01-02
    "ACO_ID",
    "performance_year",

    # 03
    "N_AB",

    # 04-23
    "P_EDV_Vis",
    "P_EDV_Vis_HOSP",
    "ADM",
    "ADM_S_Trm",
    "ADM_L_Trm",
    "ADM_Rehab",
    "ADM_Psych",
    "P_SNF_ADM",
    "SNF_LOS",
    "SNF_PayperStay",
    "chf_adm",
    "copd_adm",
    "prov_Rate_1000",
    "P_CT_VIS",
    "P_MRI_VIS",
    "P_EM_Total",
    "P_EM_PCP_Vis",
    "P_EM_SP_Vis",
    "P_Nurse_Vis",
    "P_FQHC_RHC_Vis",

    # 24-27
    "ed_visits_per_beneficiary",
    "hospital_ed_visits_per_beneficiary",
    "admissions_per_beneficiary",
    "snf_admissions_per_beneficiary",

    # 28-31
    "short_term_admission_share",
    "long_term_admission_share",
    "rehab_admission_share",
    "psych_admission_share",

    # 32-35
    "pcp_visit_share",
    "specialist_visit_share",
    "nurse_visit_share",
    "fqhc_rhc_visit_share",

    # 36-38
    "ct_visits_per_beneficiary",
    "mri_visits_per_beneficiary",
    "advanced_imaging_per_beneficiary",

    # 39-42
    "emergency_utilization_rate",
    "admission_rate_per_1000",
    "chf_admission_rate_per_1000",
    "copd_admission_rate_per_1000",

    # 43
    "snf_pay_per_admission",

    # 44
    "provider_rate_per_1000",

    # 45-49
    "ed_intensity",
    "admission_intensity",
    "em_visit_intensity",
    "ct_intensity",
    "mri_intensity",

    # 50-54
    "previous_utilization_year",
    "previous_ed_visits_per_beneficiary",
    "previous_admissions_per_beneficiary",
    "previous_em_visit_intensity",
    "previous_advanced_imaging_per_beneficiary",

    # 55-58
    "ed_utilization_change_yoy",
    "admission_change_yoy",
    "em_utilization_change_yoy",
    "advanced_imaging_change_yoy",

    # 59-62
    "utilization_score",
    "utilization_category",
    "high_utilization_flag",
    "low_utilization_flag",
]


# ============================================================
# INTEGER COLUMNS
# These MUST be sent as integers / None
# ============================================================

INTEGER_COLUMNS = [

    "performance_year",
    "N_AB",
    "previous_utilization_year",

]


# ============================================================
# NUMERIC SOURCE COLUMNS
# ============================================================

SOURCE_NUMERIC_COLUMNS = [

    "N_AB",

    "P_EDV_Vis",
    "P_EDV_Vis_HOSP",

    "ADM",
    "ADM_S_Trm",
    "ADM_L_Trm",
    "ADM_Rehab",
    "ADM_Psych",

    "P_SNF_ADM",
    "SNF_LOS",
    "SNF_PayperStay",

    "chf_adm",
    "copd_adm",

    "prov_Rate_1000",

    "P_CT_VIS",
    "P_MRI_VIS",

    "P_EM_Total",
    "P_EM_PCP_Vis",
    "P_EM_SP_Vis",

    "P_Nurse_Vis",
    "P_FQHC_RHC_Vis",
]


# ============================================================
# FETCH ALL ROWS
# ============================================================

def fetch_all_rows(table_name, columns):

    print()
    print("=" * 70)
    print(f"FETCHING: {table_name}")
    print("=" * 70)

    all_rows = []
    offset = 0

    while True:

        try:

            response = (
                supabase
                .table(table_name)
                .select(",".join(columns))
                .range(
                    offset,
                    offset + PAGE_SIZE - 1
                )
                .execute()
            )

        except Exception as e:

            print()
            print("ERROR WHILE RETRIEVING DATA")
            print(e)
            sys.exit(1)

        rows = response.data

        if not rows:
            break

        all_rows.extend(rows)

        print(
            f"{table_name}: retrieved "
            f"{len(all_rows)} rows"
        )

        if len(rows) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

        time.sleep(0.1)

    print(
        f"Total rows retrieved: "
        f"{len(all_rows)}"
    )

    return pd.DataFrame(all_rows)


# ============================================================
# SAFE NUMERIC CONVERSION
# ============================================================

def convert_numeric_columns(df):

    for col in SOURCE_NUMERIC_COLUMNS:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


# ============================================================
# SAFE INTEGER CONVERSION
# ============================================================

def convert_integer_columns(df):

    print()
    print("Converting integer columns...")

    # --------------------------------------------------------
    # performance_year
    # --------------------------------------------------------

    df["performance_year"] = pd.to_numeric(
        df["performance_year"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # N_AB
    # --------------------------------------------------------

    df["N_AB"] = pd.to_numeric(
        df["N_AB"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Convert to nullable integer
    #
    # This prevents:
    # 2020.0
    #
    # from reaching Supabase.
    # --------------------------------------------------------

    df["performance_year"] = (
        df["performance_year"]
        .round()
        .astype("Int64")
    )

    df["N_AB"] = (
        df["N_AB"]
        .round()
        .astype("Int64")
    )

    return df


# ============================================================
# SAFE DIVISION
# ============================================================

def safe_divide(numerator, denominator):

    numerator = pd.to_numeric(
        numerator,
        errors="coerce"
    )

    denominator = pd.to_numeric(
        denominator,
        errors="coerce"
    )

    result = np.where(
        denominator > 0,
        numerator / denominator,
        np.nan
    )

    return result


# ============================================================
# CREATE UTILIZATION FEATURES
# ============================================================

def create_utilization_features(df):

    print()
    print("=" * 70)
    print("CREATING ACO-LEVEL UTILIZATION FEATURES")
    print("=" * 70)

    # ========================================================
    # BASIC CLEANING
    # ========================================================

    df["ACO_ID"] = (
        df["ACO_ID"]
        .astype(str)
        .str.strip()
    )

    df = convert_numeric_columns(df)

    df = convert_integer_columns(df)

    # ========================================================
    # REMOVE INVALID KEYS
    # ========================================================

    before = len(df)

    df = df[
        df["ACO_ID"].notna()
        &
        df["performance_year"].notna()
    ].copy()

    removed = before - len(df)

    if removed > 0:

        print(
            f"Removed {removed} rows "
            f"with invalid ACO_ID/year."
        )

    # ========================================================
    # DUPLICATE ACO-YEAR CHECK
    # ========================================================

    duplicate_count = (
        df
        .duplicated(
            subset=[
                "ACO_ID",
                "performance_year"
            ]
        )
        .sum()
    )

    print(
        f"Duplicate ACO-Year rows before "
        f"processing: {duplicate_count}"
    )

    if duplicate_count > 0:

        print(
            "WARNING: Duplicate ACO-Year rows "
            "detected."
        )

        df = (
            df
            .sort_values(
                [
                    "ACO_ID",
                    "performance_year"
                ]
            )
            .drop_duplicates(
                subset=[
                    "ACO_ID",
                    "performance_year"
                ],
                keep="first"
            )
        )

    # ========================================================
    # SORT
    # ========================================================

    df = (
        df
        .sort_values(
            [
                "ACO_ID",
                "performance_year"
            ]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # BASIC UTILIZATION
    # ========================================================

    df["ed_visits_per_beneficiary"] = safe_divide(
        df["P_EDV_Vis"],
        df["N_AB"]
    )

    df["hospital_ed_visits_per_beneficiary"] = safe_divide(
        df["P_EDV_Vis_HOSP"],
        df["N_AB"]
    )

    df["admissions_per_beneficiary"] = safe_divide(
        df["ADM"],
        df["N_AB"]
    )

    df["snf_admissions_per_beneficiary"] = safe_divide(
        df["P_SNF_ADM"],
        df["N_AB"]
    )

    # ========================================================
    # ADMISSION MIX
    # ========================================================

    df["short_term_admission_share"] = safe_divide(
        df["ADM_S_Trm"],
        df["ADM"]
    )

    df["long_term_admission_share"] = safe_divide(
        df["ADM_L_Trm"],
        df["ADM"]
    )

    df["rehab_admission_share"] = safe_divide(
        df["ADM_Rehab"],
        df["ADM"]
    )

    df["psych_admission_share"] = safe_divide(
        df["ADM_Psych"],
        df["ADM"]
    )

    # ========================================================
    # VISIT MIX
    # ========================================================

    df["pcp_visit_share"] = safe_divide(
        df["P_EM_PCP_Vis"],
        df["P_EM_Total"]
    )

    df["specialist_visit_share"] = safe_divide(
        df["P_EM_SP_Vis"],
        df["P_EM_Total"]
    )

    df["nurse_visit_share"] = safe_divide(
        df["P_Nurse_Vis"],
        df["P_EM_Total"]
    )

    df["fqhc_rhc_visit_share"] = safe_divide(
        df["P_FQHC_RHC_Vis"],
        df["P_EM_Total"]
    )

    # ========================================================
    # IMAGING
    # ========================================================

    df["ct_visits_per_beneficiary"] = safe_divide(
        df["P_CT_VIS"],
        df["N_AB"]
    )

    df["mri_visits_per_beneficiary"] = safe_divide(
        df["P_MRI_VIS"],
        df["N_AB"]
    )

    df["advanced_imaging_per_beneficiary"] = (
        df["ct_visits_per_beneficiary"]
        +
        df["mri_visits_per_beneficiary"]
    )

    # ========================================================
    # RATES
    # ========================================================

    df["emergency_utilization_rate"] = (
        safe_divide(
            df["P_EDV_Vis"],
            df["N_AB"]
        )
        * 1000
    )

    df["admission_rate_per_1000"] = (
        safe_divide(
            df["ADM"],
            df["N_AB"]
        )
        * 1000
    )

    df["chf_admission_rate_per_1000"] = (
        safe_divide(
            df["chf_adm"],
            df["N_AB"]
        )
        * 1000
    )

    df["copd_admission_rate_per_1000"] = (
        safe_divide(
            df["copd_adm"],
            df["N_AB"]
        )
        * 1000
    )

    # ========================================================
    # SNF
    # ========================================================

    df["snf_pay_per_admission"] = safe_divide(
        df["SNF_PayperStay"],
        df["P_SNF_ADM"]
    )

    # ========================================================
    # PROVIDER RATE
    # ========================================================

    df["provider_rate_per_1000"] = (
        df["prov_Rate_1000"]
    )

    # ========================================================
    # INTENSITY FEATURES
    # ========================================================

    df["ed_intensity"] = safe_divide(
        df["P_EDV_Vis"],
        df["N_AB"]
    )

    df["admission_intensity"] = safe_divide(
        df["ADM"],
        df["N_AB"]
    )

    df["em_visit_intensity"] = safe_divide(
        df["P_EM_Total"],
        df["N_AB"]
    )

    df["ct_intensity"] = safe_divide(
        df["P_CT_VIS"],
        df["N_AB"]
    )

    df["mri_intensity"] = safe_divide(
        df["P_MRI_VIS"],
        df["N_AB"]
    )

    # ========================================================
    # PREVIOUS YEAR FEATURES
    # ========================================================

    grouped = df.groupby(
        "ACO_ID",
        group_keys=False
    )

    df["previous_utilization_year"] = (
        grouped["performance_year"]
        .shift(1)
    )

    df["previous_ed_visits_per_beneficiary"] = (
        grouped["ed_visits_per_beneficiary"]
        .shift(1)
    )

    df["previous_admissions_per_beneficiary"] = (
        grouped["admissions_per_beneficiary"]
        .shift(1)
    )

    df["previous_em_visit_intensity"] = (
        grouped["em_visit_intensity"]
        .shift(1)
    )

    df["previous_advanced_imaging_per_beneficiary"] = (
        grouped["advanced_imaging_per_beneficiary"]
        .shift(1)
    )

    # ========================================================
    # YOY CHANGE
    # ========================================================

    df["ed_utilization_change_yoy"] = safe_divide(
        (
            df["ed_visits_per_beneficiary"]
            -
            df["previous_ed_visits_per_beneficiary"]
        ),
        df["previous_ed_visits_per_beneficiary"]
    )

    df["admission_change_yoy"] = safe_divide(
        (
            df["admissions_per_beneficiary"]
            -
            df["previous_admissions_per_beneficiary"]
        ),
        df["previous_admissions_per_beneficiary"]
    )

    df["em_utilization_change_yoy"] = safe_divide(
        (
            df["em_visit_intensity"]
            -
            df["previous_em_visit_intensity"]
        ),
        df["previous_em_visit_intensity"]
    )

    df["advanced_imaging_change_yoy"] = safe_divide(
        (
            df["advanced_imaging_per_beneficiary"]
            -
            df["previous_advanced_imaging_per_beneficiary"]
        ),
        df["previous_advanced_imaging_per_beneficiary"]
    )

    # ========================================================
    # UTILIZATION SCORE
    # ========================================================

    score_components = [

        "ed_visits_per_beneficiary",
        "admissions_per_beneficiary",
        "advanced_imaging_per_beneficiary",
        "em_visit_intensity",
        "snf_admissions_per_beneficiary",

    ]

    percentile_columns = []

    for col in score_components:

        percentile_col = (
            f"{col}_percentile"
        )

        df[percentile_col] = (
            df[col]
            .rank(
                pct=True,
                method="average"
            )
        )

        percentile_columns.append(
            percentile_col
        )

    df["utilization_score"] = (
        df[percentile_columns]
        .mean(axis=1)
        * 100
    )

    # ========================================================
    # CATEGORY
    # ========================================================

    df["utilization_category"] = pd.cut(
        df["utilization_score"],
        bins=[
            -np.inf,
            33.33,
            66.67,
            np.inf
        ],
        labels=[
            "Low",
            "Moderate",
            "High"
        ]
    )

    # ========================================================
    # FLAGS
    # ========================================================

    df["high_utilization_flag"] = (
        df["utilization_score"] >= 66.67
    )

    df["low_utilization_flag"] = (
        df["utilization_score"] <= 33.33
    )

    # ========================================================
    # FINAL COLUMN SELECTION
    # ========================================================

    result = df[
        FINAL_COLUMNS
    ].copy()

    # ========================================================
    # CLEAN FLOAT VALUES
    # ========================================================

    result = result.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ========================================================
    # IMPORTANT:
    # Convert integer columns AFTER all calculations
    #
    # This fixes:
    #
    # 2020.0
    #
    # -> 2020
    # ========================================================

    result["performance_year"] = (
        pd.to_numeric(
            result["performance_year"],
            errors="coerce"
        )
        .round()
        .astype("Int64")
    )

    result["N_AB"] = (
        pd.to_numeric(
            result["N_AB"],
            errors="coerce"
        )
        .round()
        .astype("Int64")
    )

    result["previous_utilization_year"] = (
        pd.to_numeric(
            result["previous_utilization_year"],
            errors="coerce"
        )
        .round()
        .astype("Int64")
    )

    return result


# ============================================================
# VALIDATE DATAFRAME BEFORE INSERT
# ============================================================

def validate_before_insert(df):

    print()
    print("=" * 70)
    print("PRE-INSERT DATA VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Column count
    # --------------------------------------------------------

    if len(df.columns) != 62:

        print(
            f"ERROR: Expected 62 columns "
            f"but found {len(df.columns)}."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Exact column order
    # --------------------------------------------------------

    if list(df.columns) != FINAL_COLUMNS:

        print(
            "ERROR: Feature column order does not "
            "match expected schema."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    duplicates = (
        df
        .duplicated(
            subset=[
                "ACO_ID",
                "performance_year"
            ]
        )
        .sum()
    )

    if duplicates > 0:

        print(
            f"ERROR: {duplicates} duplicate "
            f"ACO-Year rows."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Integer validation
    # --------------------------------------------------------

    print()
    print("INTEGER COLUMN CHECK")

    for col in INTEGER_COLUMNS:

        bad_values = []

        for value in df[col]:

            if pd.isna(value):
                continue

            try:

                numeric_value = float(value)

                if not numeric_value.is_integer():

                    bad_values.append(value)

            except Exception:

                bad_values.append(value)

        if bad_values:

            print(
                f"ERROR: Invalid integer values "
                f"in {col}:"
            )

            print(
                bad_values[:10]
            )

            sys.exit(1)

        print(
            f"✓ {col}: valid"
        )

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    if np.isinf(
        numeric_df.astype(float)
    ).any().any():

        print(
            "ERROR: Infinite values remain."
        )

        sys.exit(1)

    print(
        "✓ No infinite values"
    )

    # --------------------------------------------------------
    # Required key check
    # --------------------------------------------------------

    if df["ACO_ID"].isna().any():

        print(
            "ERROR: ACO_ID contains NULL."
        )

        sys.exit(1)

    if df["performance_year"].isna().any():

        print(
            "ERROR: performance_year contains NULL."
        )

        sys.exit(1)

    print(
        "✓ ACO_ID and performance_year valid"
    )

    print()
    print(
        "PRE-INSERT VALIDATION PASSED"
    )


# ============================================================
# CONVERT DATAFRAME TO SUPABASE-SAFE RECORDS
# ============================================================

def dataframe_to_safe_records(df):

    print()
    print("=" * 70)
    print("CONVERTING DATA TO JSON-SAFE RECORDS")
    print("=" * 70)

    records = []

    for _, row in df.iterrows():

        record = {}

        for col in df.columns:

            value = row[col]

            # ------------------------------------------------
            # NULL
            # ------------------------------------------------

            if pd.isna(value):

                record[col] = None
                continue

            # ------------------------------------------------
            # INTEGER COLUMNS
            # ------------------------------------------------

            if col in INTEGER_COLUMNS:

                try:

                    integer_value = int(
                        round(float(value))
                    )

                    record[col] = integer_value

                except Exception:

                    print()
                    print(
                        f"ERROR: Could not convert "
                        f"{col} value {value!r} "
                        f"to integer."
                    )

                    sys.exit(1)

                continue

            # ------------------------------------------------
            # Boolean
            # ------------------------------------------------

            if isinstance(
                value,
                (bool, np.bool_)
            ):

                record[col] = bool(value)
                continue

            # ------------------------------------------------
            # NumPy integer
            # ------------------------------------------------

            if isinstance(
                value,
                np.integer
            ):

                record[col] = int(value)
                continue

            # ------------------------------------------------
            # NumPy float
            # ------------------------------------------------

            if isinstance(
                value,
                np.floating
            ):

                value = float(value)

                if not np.isfinite(value):

                    record[col] = None

                else:

                    record[col] = value

                continue

            # ------------------------------------------------
            # Python float
            # ------------------------------------------------

            if isinstance(
                value,
                float
            ):

                if not np.isfinite(value):

                    record[col] = None

                else:

                    record[col] = value

                continue

            # ------------------------------------------------
            # String
            # ------------------------------------------------

            if isinstance(
                value,
                str
            ):

                record[col] = value
                continue

            # ------------------------------------------------
            # Everything else
            # ------------------------------------------------

            record[col] = value

        records.append(record)

    return records


# ============================================================
# INSERT DATA
# ============================================================

def insert_data(df):

    print()
    print("=" * 70)
    print("INSERTING UTILIZATION FEATURES INTO SUPABASE")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_before_insert(df)

    # --------------------------------------------------------
    # Convert to safe records
    # --------------------------------------------------------

    records = dataframe_to_safe_records(
        df
    )

    print()
    print(
        f"Clean records ready: "
        f"{len(records)}"
    )

    # --------------------------------------------------------
    # Explicitly verify integer values
    # --------------------------------------------------------

    print()
    print("FINAL INTEGER VALUE CHECK")

    for col in INTEGER_COLUMNS:

        sample_values = [
            record[col]
            for record in records[:10]
            if record[col] is not None
        ]

        print(
            f"{col}: {sample_values}"
        )

    # --------------------------------------------------------
    # Verify no invalid floats
    # --------------------------------------------------------

    for i, record in enumerate(records):

        for key, value in record.items():

            if isinstance(value, float):

                if not np.isfinite(value):

                    print()
                    print(
                        "ERROR: Invalid float found."
                    )

                    print(
                        f"Row: {i}"
                    )

                    print(
                        f"Column: {key}"
                    )

                    print(
                        f"Value: {value}"
                    )

                    sys.exit(1)

    print()
    print(
        "✓ No invalid JSON float values"
    )

    # --------------------------------------------------------
    # Insert in batches
    # --------------------------------------------------------

    total = len(records)

    for start in range(
        0,
        total,
        INSERT_BATCH_SIZE
    ):

        end = min(
            start + INSERT_BATCH_SIZE,
            total
        )

        batch = records[
            start:end
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
                "ERROR INSERTING DATA"
            )

            print(e)

            print()
            print(
                f"Failed batch: "
                f"{start + 1} - {end}"
            )

            sys.exit(1)

        print(
            f"Inserted/upserted "
            f"{end} / {total}"
        )

        time.sleep(0.1)

    print()
    print(
        "INSERT COMPLETE"
    )


# ============================================================
# VALIDATE RESULT
# ============================================================

def validate_result(df):

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print(
        f"Rows created: {len(df)}"
    )

    print(
        f"Unique ACOs: "
        f"{df['ACO_ID'].nunique()}"
    )

    years = sorted(
        df["performance_year"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    print(
        f"Years: {years}"
    )

    duplicates = (
        df
        .duplicated(
            subset=[
                "ACO_ID",
                "performance_year"
            ]
        )
        .sum()
    )

    print(
        f"Duplicate ACO-Year rows: "
        f"{duplicates}"
    )

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    print()
    print(
        "Utilization score statistics:"
    )

    print(
        df[
            "utilization_score"
        ].describe()
    )

    # --------------------------------------------------------
    # Categories
    # --------------------------------------------------------

    print()
    print(
        "Utilization categories:"
    )

    print(
        df[
            "utilization_category"
        ]
        .value_counts(
            dropna=False
        )
    )

    # --------------------------------------------------------
    # Missing
    # --------------------------------------------------------

    print()
    print(
        "Missing-value percentage:"
    )

    important_columns = [

        "ed_visits_per_beneficiary",
        "admissions_per_beneficiary",
        "advanced_imaging_per_beneficiary",
        "em_visit_intensity",
        "utilization_score",

    ]

    for col in important_columns:

        missing_pct = (
            df[col]
            .isna()
            .mean()
            * 100
        )

        print(
            f"{col:45s} "
            f"{missing_pct:.2f}% missing"
        )


# ============================================================
# PRINT EXACT COLUMNS
# ============================================================

def print_feature_columns(df):

    print()
    print("=" * 70)
    print("EXACT FEATURE TABLE COLUMNS")
    print("=" * 70)

    for i, col in enumerate(
        df.columns,
        start=1
    ):

        print(
            f"{i:02d}. {col}"
        )

    print()
    print(
        f"TOTAL COLUMNS: "
        f"{len(df.columns)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "STEP 7B — ACO UTILIZATION "
        "FEATURE ENGINEERING"
    )
    print("=" * 70)

    # ========================================================
    # FETCH
    # ========================================================

    source_df = fetch_all_rows(
        SOURCE_TABLE,
        SOURCE_FEATURES
    )

    if source_df.empty:

        print()
        print(
            "ERROR: No rows retrieved."
        )

        sys.exit(1)

    # ========================================================
    # CREATE FEATURES
    # ========================================================

    feature_df = create_utilization_features(
        source_df
    )

    # ========================================================
    # PRINT COLUMNS
    # ========================================================

    print_feature_columns(
        feature_df
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    validate_result(
        feature_df
    )

    # ========================================================
    # READY
    # ========================================================

    print()
    print("=" * 70)
    print("READY TO WRITE TO SUPABASE")
    print("=" * 70)

    print(
        f"Target table: "
        f"{TARGET_TABLE}"
    )

    print(
        f"Rows: "
        f"{len(feature_df)}"
    )

    print(
        f"Columns: "
        f"{len(feature_df.columns)}"
    )

    # ========================================================
    # INSERT
    # ========================================================

    insert_data(
        feature_df
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 7B COMPLETE")
    print("=" * 70)

    print(
        f"Created/updated table: "
        f"{TARGET_TABLE}"
    )

    print(
        "Source table was not modified."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()