"""
Train the four prediction heads (facility, usage day-of-week, usage hour,
lead time / notification offset) on the chronological training split and
persist all artifacts needed for evaluation and future prediction.

Run: python -m src.train
"""
import os
import json
import joblib
import numpy as np

from src.features import load_raw, build_features, chronological_split, TARGET_LEADTIME
from src.models import (
    DesignMatrix, make_facility_model, make_dow_model, make_hour_model, make_leadtime_model,
)

DATA_PATH = "data/bookings.csv"
MODELS_DIR = "models"
DATA_DIR = "data"


def main():
    if not os.path.exists(DATA_PATH):
        from src.generate_dataset import generate
        os.makedirs(DATA_DIR, exist_ok=True)
        generate().to_csv(DATA_PATH, index=False)

    raw = load_raw(DATA_PATH)
    feat = build_features(raw)
    train, test, cutoff = chronological_split(feat, test_frac=0.18)

    os.makedirs(MODELS_DIR, exist_ok=True)
    train.to_csv(f"{DATA_DIR}/train.csv", index=False)
    test.to_csv(f"{DATA_DIR}/test.csv", index=False)

    # Stage 1: facility model uses only history/context features (no facility leakage).
    dm_facility = DesignMatrix()
    X_train_facility = dm_facility.fit_transform(train)
    facility_model = make_facility_model().fit(X_train_facility, train["facility"])

    # Stage 2: day/hour/lead-time models additionally condition on facility_context.
    # At TRAIN time this is the actual (known) facility for that historical booking.
    train = train.copy()
    train["facility_context"] = train["facility"]
    dm_time = DesignMatrix(extra_categorical=["facility_context"])
    X_train_time = dm_time.fit_transform(train)

    dow_model = make_dow_model().fit(X_train_time, train["usage_dow"])
    hour_model = make_hour_model().fit(X_train_time, train["usage_hour"])
    leadtime_model = make_leadtime_model().fit(X_train_time, np.log1p(train[TARGET_LEADTIME]))

    joblib.dump(dm_facility, f"{MODELS_DIR}/design_matrix_facility.joblib", compress=3)
    joblib.dump(dm_time, f"{MODELS_DIR}/design_matrix_time.joblib", compress=3)
    joblib.dump(facility_model, f"{MODELS_DIR}/facility_model.joblib", compress=3)
    joblib.dump(dow_model, f"{MODELS_DIR}/dow_model.joblib", compress=3)
    joblib.dump(hour_model, f"{MODELS_DIR}/hour_model.joblib", compress=3)
    joblib.dump(leadtime_model, f"{MODELS_DIR}/leadtime_model.joblib", compress=3)

    manifest = dict(
        train_rows=len(train), test_rows=len(test),
        cutoff_timestamp=str(cutoff),
        n_features_facility_stage=X_train_facility.shape[1],
        n_features_time_stage=X_train_time.shape[1],
        trained_at=str(np.datetime64("now")),
    )
    with open(f"{MODELS_DIR}/manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Trained on {len(train)} rows, held out {len(test)} rows after {cutoff}.")
    print(f"Feature matrix width: facility stage={X_train_facility.shape[1]}, time stage={X_train_time.shape[1]}")
    print(f"Artifacts written to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
