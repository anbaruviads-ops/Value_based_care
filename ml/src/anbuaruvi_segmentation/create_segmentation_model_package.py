import os
import json
import pickle
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

OLD_MODEL_PATH = os.path.join(
    "models",
    "aco_segmentation_kmeans.pkl"
)

NEW_MODEL_PATH = os.path.join(
    "models",
    "model.pkl"
)

FEATURES_PATH = os.path.join(
    "models",
    "features.json"
)

MODEL_INFO_PATH = os.path.join(
    "models",
    "model_info.json"
)


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
# LOAD EXISTING TRAINED MODEL
# ============================================================

print("=" * 70)
print("CREATING SEGMENTATION MODEL PACKAGE")
print("=" * 70)

print()
print("Loading existing trained model...")

if not os.path.exists(OLD_MODEL_PATH):

    print(
        f"ERROR: Existing model not found: {OLD_MODEL_PATH}"
    )

    raise SystemExit(1)


with open(
    OLD_MODEL_PATH,
    "rb"
) as file:

    old_package = pickle.load(file)


print("Existing model loaded successfully.")


# ============================================================
# EXTRACT EXISTING MODEL + SCALER
# ============================================================

if "model" not in old_package:

    print(
        "ERROR: Existing model file does not contain 'model'."
    )

    raise SystemExit(1)


if "scaler" not in old_package:

    print(
        "ERROR: Existing model file does not contain 'scaler'."
    )

    raise SystemExit(1)


kmeans_model = old_package["model"]
scaler = old_package["scaler"]


print()
print("Existing components found:")
print(
    f"Model  : {type(kmeans_model).__name__}"
)
print(
    f"Scaler : {type(scaler).__name__}"
)


# ============================================================
# CREATE COMPLETE PIPELINE
# ============================================================

print()
print("Creating complete preprocessing + model pipeline...")


pipeline = Pipeline(
    steps=[
        (
            "scaler",
            scaler
        ),
        (
            "kmeans",
            kmeans_model
        )
    ]
)


# ============================================================
# SAVE MODEL.PKL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(
    pipeline,
    NEW_MODEL_PATH
)


print()
print(
    f"SUCCESS: Complete pipeline saved to:"
)
print(
    f"  {NEW_MODEL_PATH}"
)


# ============================================================
# SAVE FEATURES.JSON
# ============================================================

features_data = {
    "features": FEATURES
}


with open(
    FEATURES_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        features_data,
        file,
        indent=2
    )


print()
print(
    f"SUCCESS: Features saved to:"
)
print(
    f"  {FEATURES_PATH}"
)


# ============================================================
# SAVE MODEL_INFO.JSON
# ============================================================

model_info = {
    "model_name": "aco_segmentation",
    "version": "1.0",
    "model_type": "KMeans",
    "preprocessing": "StandardScaler",
    "pipeline": True,
    "n_clusters": int(
        kmeans_model.n_clusters
    ),
    "features_count": len(FEATURES),
    "features_file": "features.json",
    "purpose": "ACO Performance Segmentation",
    "segments": [
        "High Performing",
        "Moderate",
        "Needs Attention"
    ]
}


with open(
    MODEL_INFO_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        model_info,
        file,
        indent=2
    )


print()
print(
    f"SUCCESS: Model information saved to:"
)
print(
    f"  {MODEL_INFO_PATH}"
)


# ============================================================
# VERIFY PIPELINE
# ============================================================

print()
print("=" * 70)
print("VERIFYING MODEL PACKAGE")
print("=" * 70)


loaded_pipeline = joblib.load(
    NEW_MODEL_PATH
)


print()
print(
    "Pipeline loaded successfully."
)

print()
print("Pipeline steps:")

for name, step in loaded_pipeline.steps:

    print(
        f"  {name}: {type(step).__name__}"
    )


print()
print("Expected features:")
print(
    len(FEATURES)
)

print()
print("=" * 70)
print("MODEL PACKAGE CREATION COMPLETE")
print("=" * 70)

print()
print("Created files:")

print(
    f"1. {NEW_MODEL_PATH}"
)

print(
    f"2. {FEATURES_PATH}"
)

print(
    f"3. {MODEL_INFO_PATH}"
)

print()
print(
    "IMPORTANT:"
)

print(
    "No K-Means retraining was performed."
)

print(
    "No Supabase tables were modified."
)

print(
    "Existing segmentation results remain unchanged."
)