"""
Forward-looking prediction: for every resident, predict their NEXT booking
(facility, usage day-of-week, usage hour, and when to send a nudge) as of
"today" (the day after the last booking in the dataset), using their full
booking history to date.

This demonstrates the standalone prediction workflow (as opposed to
evaluate.py, which scores predictions against already-known historical
outcomes on the held-out test set).

Run: python -m src.predict
"""
import joblib
import numpy as np
import pandas as pd

from src.features import load_raw, build_features, DAY_NAMES
from src.evaluate import fmt_hhmm

MODELS_DIR = "models"
DATA_PATH = "data/bookings.csv"
OUT_PATH = "outputs/next_booking_predictions.csv"


def build_query_rows(raw: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """One placeholder 'next booking' row per resident, dated `as_of`, so that
    build_features() computes each resident's history-to-date features for it."""
    last_known = raw.sort_values("booking_timestamp").groupby("resident_id").tail(1)
    query = last_known.copy()
    query["booking_id"] = "QUERY-" + query["resident_id"]
    query["booking_timestamp"] = as_of
    query["usage_timestamp"] = as_of  # placeholder only; usage_dow/hour derived from this are targets, unused as features
    return query


def main():
    raw = load_raw(DATA_PATH)
    as_of = raw["booking_timestamp"].max() + pd.Timedelta(days=1)
    query_rows = build_query_rows(raw, as_of)

    combined = pd.concat([raw, query_rows], ignore_index=True)
    feat = build_features(combined)
    query_feat = feat[feat["booking_id"].str.startswith("QUERY-")].reset_index(drop=True)

    dm_facility = joblib.load(f"{MODELS_DIR}/design_matrix_facility.joblib")
    dm_time = joblib.load(f"{MODELS_DIR}/design_matrix_time.joblib")
    facility_model = joblib.load(f"{MODELS_DIR}/facility_model.joblib")
    dow_model = joblib.load(f"{MODELS_DIR}/dow_model.joblib")
    hour_model = joblib.load(f"{MODELS_DIR}/hour_model.joblib")
    leadtime_model = joblib.load(f"{MODELS_DIR}/leadtime_model.joblib")

    X_facility = dm_facility.transform(query_feat)
    pred_facility = facility_model.predict(X_facility)

    query_feat = query_feat.copy()
    query_feat["facility_context"] = pred_facility
    X_time = dm_time.transform(query_feat)
    pred_dow = dow_model.predict(X_time)
    pred_hour = hour_model.predict(X_time)
    pred_leadtime = np.expm1(leadtime_model.predict(X_time))

    rows = []
    for i, row in query_feat.iterrows():
        nudge_hours_before = float(pred_leadtime[i]) % 168  # wrap within a week for display
        nudge_dow = (int(pred_dow[i]) * 24 + float(pred_hour[i]) - nudge_hours_before) / 24
        nudge_dow_idx = int(np.floor(nudge_dow)) % 7
        nudge_hour = (nudge_dow % 1) * 24
        rows.append({
            "resident_id": row["resident_id"],
            "as_of_date": as_of.date().isoformat(),
            "bookings_in_history": int(row["n_prior_bookings"]),
            "predicted_next_facility": pred_facility[i],
            "predicted_usage_day": DAY_NAMES[int(pred_dow[i])],
            "predicted_usage_time": fmt_hhmm(pred_hour[i]),
            "predicted_lead_time_hours": round(float(pred_leadtime[i]), 1),
            "predicted_nudge_day": DAY_NAMES[nudge_dow_idx],
            "predicted_nudge_time": fmt_hhmm(nudge_hour),
        })
    out = pd.DataFrame(rows).sort_values("resident_id")
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote next-booking predictions for {len(out)} residents -> {OUT_PATH}")
    print(out.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
