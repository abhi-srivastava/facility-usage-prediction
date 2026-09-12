"""
Evaluate the trained models on the held-out chronological test set and
produce:
  - outputs/metrics.json            summary + per-output + error analysis
  - outputs/predictions_review.csv  per-booking predicted-vs-actual table
  - outputs/predictions_review.xlsx same, as a formatted spreadsheet

Run: python -m src.evaluate
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, median_absolute_error

from src.features import load_raw, build_features, DAY_NAMES, TARGET_LEADTIME

MODELS_DIR = "models"
DATA_DIR = "data"
OUT_DIR = "outputs"
HOUR_TOLERANCE = 0  # usage hours are discrete facility slots -> exact match


def leadtime_tolerance_hours(actual_leadtime_hours: np.ndarray) -> np.ndarray:
    """A nudge just needs to land close to the ideal moment RELATIVE to how far
    ahead that booking is typically made: +/-3h for a same-day gym slot is
    strict but fair, while +/-3h for a two-week-ahead clubhouse booking is
    unreasonably strict -- so the window scales with the lead time itself,
    floored at 3 hours."""
    return np.maximum(3.0, 0.15 * actual_leadtime_hours)


def circular_dow_distance(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 7 - d)


def fmt_hhmm(hour_float):
    hour_float = float(hour_float) % 24
    h = int(hour_float)
    m = int(round((hour_float - h) * 60)) % 60
    return f"{h:02d}:{m:02d}"


def anchor_datetime(actual_usage_ts: pd.Timestamp, target_dow: int, hour_float: float):
    """Place a (day-of-week, hour) pair in the same ISO week as the actual usage
    timestamp, purely so predictions can be shown as readable day/time strings
    in the review table. Scoring never uses this -- it compares the raw
    dow/hour/lead-time numbers directly."""
    delta_days = target_dow - actual_usage_ts.dayofweek
    anchored_date = actual_usage_ts.normalize() + pd.Timedelta(days=int(delta_days))
    return anchored_date + pd.Timedelta(hours=float(hour_float))


def main():
    dm_facility = joblib.load(f"{MODELS_DIR}/design_matrix_facility.joblib")
    dm_time = joblib.load(f"{MODELS_DIR}/design_matrix_time.joblib")
    facility_model = joblib.load(f"{MODELS_DIR}/facility_model.joblib")
    dow_model = joblib.load(f"{MODELS_DIR}/dow_model.joblib")
    hour_model = joblib.load(f"{MODELS_DIR}/hour_model.joblib")
    leadtime_model = joblib.load(f"{MODELS_DIR}/leadtime_model.joblib")

    test = pd.read_csv(f"{DATA_DIR}/test.csv", parse_dates=["booking_timestamp", "usage_timestamp", "move_in_date"])
    X_test_facility = dm_facility.transform(test)
    pred_facility = facility_model.predict(X_test_facility)
    facility_proba = facility_model.predict_proba(X_test_facility)
    facility_classes = facility_model.classes_

    # day/hour/lead-time condition on the PREDICTED facility (true future facility is unknown at inference time)
    test = test.copy()
    test["facility_context"] = pred_facility
    X_test_time = dm_time.transform(test)
    pred_dow = dow_model.predict(X_test_time)
    pred_hour = hour_model.predict(X_test_time)
    pred_leadtime = np.expm1(leadtime_model.predict(X_test_time))

    actual_facility = test["facility"].to_numpy()
    actual_dow = test["usage_dow"].to_numpy()
    actual_hour = test["usage_hour"].to_numpy()
    actual_leadtime = test[TARGET_LEADTIME].to_numpy()

    facility_match = pred_facility == actual_facility
    dow_match = circular_dow_distance(pred_dow, actual_dow) <= 0
    hour_match = np.abs(pred_hour - actual_hour) <= HOUR_TOLERANCE
    leadtime_match = np.abs(pred_leadtime - actual_leadtime) <= leadtime_tolerance_hours(actual_leadtime)

    matches = np.vstack([facility_match, dow_match, hour_match, leadtime_match]).T
    n_matches = matches.sum(axis=1)

    facility_confidence = facility_proba.max(axis=1)
    top3_idx = np.argsort(-facility_proba, axis=1)[:, :3]
    top3_facilities = facility_classes[top3_idx]
    top1_acc = float(facility_match.mean())
    top2_acc = float(np.mean([actual_facility[i] in top3_facilities[i, :2] for i in range(len(test))]))
    top3_acc = float(np.mean([actual_facility[i] in top3_facilities[i, :3] for i in range(len(test))]))

    # ---------------- metrics ----------------
    metrics = {
        "test_rows": int(len(test)),
        "cutoff_timestamp_used_for_split": None,
        "per_output": {
            "facility": {
                "accuracy": float(accuracy_score(actual_facility, pred_facility)),
                "macro_f1": float(f1_score(actual_facility, pred_facility, average="macro")),
                "top1_accuracy": top1_acc,
                "top2_accuracy": top2_acc,
                "top3_accuracy": top3_acc,
                "note": "topN = correct facility appears among the model's N highest-probability guesses",
            },
            "usage_day_of_week": {
                "exact_accuracy": float(accuracy_score(actual_dow, pred_dow)),
                "within_1_day_accuracy": float((circular_dow_distance(pred_dow, actual_dow) <= 1).mean()),
            },
            "usage_hour": {
                "exact_accuracy": float(accuracy_score(actual_hour, pred_hour)),
                "mae_hours": float(mean_absolute_error(actual_hour, pred_hour)),
            },
            "notification_time_leadtime": {
                "mae_hours": float(mean_absolute_error(actual_leadtime, pred_leadtime)),
                "median_absolute_error_hours": float(median_absolute_error(actual_leadtime, pred_leadtime)),
                "rmse_hours": float(np.sqrt(mean_squared_error(actual_leadtime, pred_leadtime))),
                "mae_log_hours": float(mean_absolute_error(np.log1p(actual_leadtime), np.log1p(pred_leadtime))),
                "match_within_tolerance_rate": float(leadtime_match.mean()),
                "tolerance_note": "match = within max(3h, 15% of actual lead time)",
            },
        },
        "overall": {
            "mean_outputs_correct_out_of_4": float(n_matches.mean()),
            "all_4_correct_rate": float((n_matches == 4).mean()),
            "at_least_2_of_4_correct_rate": float((n_matches >= 2).mean()),
            "per_output_average_accuracy": float(matches.mean()),
        },
        "baselines": {},
        "error_analysis": {},
    }

    # naive baselines for context: always predict resident's own historical mode / global mode
    train = pd.read_csv(f"{DATA_DIR}/train.csv", parse_dates=["booking_timestamp", "usage_timestamp"])
    global_top_facility = train["facility"].mode().iat[0]
    global_top_dow = train["usage_dow"].mode().iat[0]
    global_top_hour = train["usage_hour"].mode().iat[0]
    global_median_leadtime = train[TARGET_LEADTIME].median()
    metrics["baselines"]["always_predict_global_mode"] = {
        "facility_accuracy": float((actual_facility == global_top_facility).mean()),
        "dow_accuracy": float((actual_dow == global_top_dow).mean()),
        "hour_accuracy": float((actual_hour == global_top_hour).mean()),
        "leadtime_mae_hours": float(mean_absolute_error(actual_leadtime, np.full_like(actual_leadtime, global_median_leadtime))),
    }
    # resident's own last-known facility as a "persistence" baseline
    persistence_acc = float((test["last_facility"] == actual_facility).mean())
    metrics["baselines"]["persistence_last_facility_accuracy"] = persistence_acc

    # error analysis: accuracy by facility, and by history depth (cold-start vs warm)
    review_tmp = test.copy()
    review_tmp["facility_match"] = facility_match
    metrics["error_analysis"]["facility_accuracy_by_true_facility"] = (
        review_tmp.groupby("facility")["facility_match"].mean().round(3).to_dict()
    )
    depth_bucket = pd.cut(test["n_prior_bookings"], bins=[-1, 0, 3, 10, 1000],
                          labels=["cold_start_0", "warm_1_3", "warm_4_10", "warm_10plus"])
    metrics["error_analysis"]["facility_accuracy_by_history_depth"] = (
        pd.Series(facility_match).groupby(depth_bucket.values, observed=True).mean().round(3).to_dict()
    )
    # confidence-gated nudges: only act on predictions the model is confident about --
    # trades coverage (how many residents get a nudge) for precision (how often it's right)
    thresholds = [0.0, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
    confidence_gate = []
    for t in thresholds:
        gated = facility_confidence >= t
        coverage = float(gated.mean())
        precision = float(facility_match[gated].mean()) if gated.any() else None
        confidence_gate.append({
            "min_confidence": t, "coverage_rate": coverage, "facility_accuracy_when_nudged": precision,
        })
    metrics["error_analysis"]["confidence_gated_nudging"] = confidence_gate

    top_confusions = (
        pd.DataFrame({"actual": actual_facility, "predicted": pred_facility})
        .query("actual != predicted")
        .value_counts()
        .head(10)
    )
    metrics["error_analysis"]["top_facility_confusions"] = {
        f"{a} -> predicted {p}": int(c) for (a, p), c in top_confusions.items()
    }

    metrics = _nan_to_none(metrics)
    with open(f"{OUT_DIR}/metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    # ---------------- prediction review table ----------------
    rows = []
    for i, row in test.reset_index(drop=True).iterrows():
        pred_usage_dt = anchor_datetime(row["usage_timestamp"], int(pred_dow[i]), float(pred_hour[i]))
        pred_nudge_dt = pred_usage_dt - pd.Timedelta(hours=float(pred_leadtime[i]))
        rows.append({
            "record_ref": row["booking_id"],
            "resident_id": row["resident_id"],
            "past_bookings_seen": int(row["n_prior_bookings"]),
            "last_facility_booked": row["last_facility"],
            "pred_facility": pred_facility[i],
            "pred_facility_confidence": round(float(facility_confidence[i]), 3),
            "pred_facility_top3": ", ".join(top3_facilities[i]),
            "pred_usage_day": DAY_NAMES[int(pred_dow[i])],
            "pred_usage_time": fmt_hhmm(pred_hour[i]),
            "pred_nudge_day": DAY_NAMES[pred_nudge_dt.dayofweek],
            "pred_nudge_time": fmt_hhmm(pred_nudge_dt.hour + pred_nudge_dt.minute / 60),
            "actual_facility": row["facility"],
            "actual_usage_day": DAY_NAMES[int(row["usage_dow"])],
            "actual_usage_time": fmt_hhmm(row["usage_timestamp"].hour + row["usage_timestamp"].minute / 60),
            "actual_booked_day": DAY_NAMES[row["booking_timestamp"].dayofweek],
            "actual_booked_time": fmt_hhmm(row["booking_timestamp"].hour + row["booking_timestamp"].minute / 60),
            "facility_match": bool(facility_match[i]),
            "usage_day_match": bool(dow_match[i]),
            "usage_hour_match": bool(hour_match[i]),
            "notification_time_match": bool(leadtime_match[i]),  # within max(3h, 15% of actual lead time)
            "matches_out_of_4": int(n_matches[i]),
            "all_match": bool(n_matches[i] == 4),
        })
    review = pd.DataFrame(rows)
    review.to_csv(f"{OUT_DIR}/predictions_review.csv", index=False)

    try:
        with pd.ExcelWriter(f"{OUT_DIR}/predictions_review.xlsx", engine="openpyxl") as writer:
            review.to_excel(writer, sheet_name="predictions_vs_actual", index=False)
            summary_df = pd.json_normalize(metrics, sep=".").T.rename(columns={0: "value"})
            summary_df.to_excel(writer, sheet_name="summary_metrics")
        _color_matches(f"{OUT_DIR}/predictions_review.xlsx")
    except Exception as e:
        print(f"(Excel export skipped: {e})")

    print(f"Evaluated {len(test)} held-out bookings.")
    print(f"Overall: {metrics['overall']['mean_outputs_correct_out_of_4']:.2f} / 4 outputs correct on average, "
          f"{metrics['overall']['all_4_correct_rate']*100:.1f}% rows fully correct.")
    print(f"Wrote outputs/metrics.json, outputs/predictions_review.csv, outputs/predictions_review.xlsx")


def _nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _color_matches(xlsx_path):
    from openpyxl import load_workbook
    from openpyxl.styles import PatternFill
    wb = load_workbook(xlsx_path)
    ws = wb["predictions_vs_actual"]
    green = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    red = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    header = [c.value for c in ws[1]]
    match_cols = [header.index(c) + 1 for c in
                  ["facility_match", "usage_day_match", "usage_hour_match", "notification_time_match", "all_match"]]
    for r in range(2, ws.max_row + 1):
        for c in match_cols:
            cell = ws.cell(row=r, column=c)
            cell.fill = green if cell.value else red
    wb.save(xlsx_path)


if __name__ == "__main__":
    main()
