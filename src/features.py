"""
Leakage-safe feature engineering.

Every feature for booking row i is computed using ONLY information that
would have been available strictly before that booking was made:
  - this resident's own prior bookings (booking_timestamp < row i's booking_timestamp)
  - community-wide prior bookings (same cutoff, across all residents)
  - static resident attributes known at move-in (household size, has_kids, move-in date)
  - the calendar context of the moment the booking was made (month/day/hour it was booked)

None of the four prediction targets (facility, usage day-of-week, usage hour,
lead time / notification offset) are used to build any feature.

The "shift(1) then expanding()" pattern is used throughout: shifting first
removes the current row from the window, so the following expanding
aggregate only ever sees strictly-prior rows.
"""
import numpy as np
import pandas as pd

FACILITY_NAMES = [
    "Gym", "Swimming Pool", "Badminton Court", "Tennis Court",
    "Squash Court", "Clubhouse", "Multipurpose Hall", "Kids Play Area",
]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_raw(path="data/bookings.csv"):
    df = pd.read_csv(path, parse_dates=["booking_timestamp", "usage_timestamp", "move_in_date"])
    return df


def _prior_expanding(series: pd.Series, agg: str):
    """Expanding aggregate over strictly-prior values (current row excluded)."""
    shifted = series.shift(1)
    if agg == "mean":
        return shifted.expanding().mean()
    if agg == "std":
        return shifted.expanding().std()
    if agg == "median":
        return shifted.expanding().median()
    if agg == "count":
        return shifted.expanding().count()
    raise ValueError(agg)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["booking_timestamp", "booking_id"]).reset_index(drop=True)

    df["usage_dow"] = df["usage_timestamp"].dt.dayofweek       # target A component (0=Mon)
    df["usage_hour"] = df["usage_timestamp"].dt.hour            # target component
    df["lead_time_hours"] = (df["usage_timestamp"] - df["booking_timestamp"]).dt.total_seconds() / 3600.0

    # ---------------- booking-time calendar context (known the instant the booking is made) ----------------
    df["booking_month"] = df["booking_timestamp"].dt.month
    df["booking_dow"] = df["booking_timestamp"].dt.dayofweek
    df["booking_hour"] = df["booking_timestamp"].dt.hour
    df["booking_is_weekend"] = (df["booking_dow"] >= 5).astype(int)
    df["days_since_movein"] = (df["booking_timestamp"] - df["move_in_date"]).dt.days.clip(lower=0)
    df["household_size"] = df["household_size"].fillna(df["household_size"].median())
    df["has_kids"] = df["has_kids"].astype(int)

    g = df.groupby("resident_id", sort=False)

    # ---------------- resident history depth / recency ----------------
    df["n_prior_bookings"] = g.cumcount()
    df["days_since_last_booking"] = (
        df["booking_timestamp"] - g["booking_timestamp"].shift(1)
    ).dt.total_seconds() / 86400.0
    df["days_since_last_booking"] = df["days_since_last_booking"].fillna(df["days_since_movein"])

    # ---------------- resident facility-preference history (proportions over prior bookings only) ----------------
    for f in FACILITY_NAMES:
        ind = (df["facility"] == f).astype(int)
        prior_count = ind.groupby(df["resident_id"]).cumsum() - ind
        df[f"pref_{f}"] = (prior_count / df["n_prior_bookings"].replace(0, np.nan)).fillna(0.0)
    df["last_facility"] = g["facility"].shift(1)

    # ---------------- resident day-of-week history ----------------
    for d in range(7):
        ind = (df["usage_dow"] == d).astype(int)
        prior_count = ind.groupby(df["resident_id"]).cumsum() - ind
        df[f"daypref_{DAY_NAMES[d]}"] = (prior_count / df["n_prior_bookings"].replace(0, np.nan)).fillna(0.0)

    # ---------------- resident hour-of-day history ----------------
    df["hour_hist_mean"] = g["usage_hour"].transform(lambda s: _prior_expanding(s, "mean"))
    df["hour_hist_std"] = g["usage_hour"].transform(lambda s: _prior_expanding(s, "std"))
    df["last_hour"] = g["usage_hour"].shift(1)

    # ---------------- resident lead-time (booking-ahead habit) history ----------------
    df["leadtime_hist_mean"] = g["lead_time_hours"].transform(lambda s: _prior_expanding(s, "mean"))
    df["leadtime_hist_median"] = g["lead_time_hours"].transform(lambda s: _prior_expanding(s, "median"))
    df["leadtime_hist_std"] = g["lead_time_hours"].transform(lambda s: _prior_expanding(s, "std"))

    # ---------------- community-wide (global) prior popularity, time-aware ----------------
    # relies on df already being sorted ascending by booking_timestamp -> row-order shift == time order
    total_prior = pd.Series(np.arange(len(df)), index=df.index)  # rows strictly before i = i
    for f in FACILITY_NAMES:
        ind = (df["facility"] == f).astype(int)
        cum_before = ind.cumsum().shift(1).fillna(0)
        df[f"global_share_{f}"] = (cum_before / total_prior.replace(0, np.nan)).fillna(
            1.0 / len(FACILITY_NAMES)
        )
    df["global_leadtime_median"] = _prior_expanding(df["lead_time_hours"], "median")

    # ---------------- fill cold-start (no personal history yet) with global fallbacks ----------------
    df["is_cold_start"] = (df["n_prior_bookings"] == 0).astype(int)
    df["hour_hist_mean"] = df["hour_hist_mean"].fillna(df["booking_hour"])
    df["hour_hist_std"] = df["hour_hist_std"].fillna(0.0)
    global_lt_fallback = df["global_leadtime_median"].fillna(df["global_leadtime_median"].median())
    for col in ["leadtime_hist_mean", "leadtime_hist_median"]:
        df[col] = df[col].fillna(global_lt_fallback)
    df["leadtime_hist_std"] = df["leadtime_hist_std"].fillna(0.0)
    df["global_leadtime_median"] = global_lt_fallback
    df["last_facility"] = df["last_facility"].fillna("NONE")
    df["last_hour"] = df["last_hour"].fillna(df["hour_hist_mean"])

    return df


FEATURE_COLUMNS_NUMERIC = (
    ["household_size", "has_kids", "days_since_movein", "n_prior_bookings",
     "days_since_last_booking", "hour_hist_mean", "hour_hist_std",
     "leadtime_hist_mean", "leadtime_hist_median", "leadtime_hist_std",
     "global_leadtime_median", "is_cold_start", "last_hour",
     "booking_month", "booking_dow", "booking_hour", "booking_is_weekend"]
    + [f"pref_{f}" for f in FACILITY_NAMES]
    + [f"daypref_{d}" for d in DAY_NAMES]
    + [f"global_share_{f}" for f in FACILITY_NAMES]
)
FEATURE_COLUMNS_CATEGORICAL = ["last_facility"]

TARGET_FACILITY = "facility"
TARGET_DOW = "usage_dow"
TARGET_HOUR = "usage_hour"
TARGET_LEADTIME = "lead_time_hours"


def chronological_split(df: pd.DataFrame, test_frac=0.18):
    """Split by booking_timestamp: the most recent test_frac of bookings become
    the held-out test set. Because features for every row only ever look at
    bookings strictly earlier than that row, this split has no leakage even
    though some 'training-period' rows for a resident may sit chronologically
    after other residents' test rows."""
    df = df.sort_values("booking_timestamp").reset_index(drop=True)
    cutoff_idx = int(len(df) * (1 - test_frac))
    cutoff_time = df.loc[cutoff_idx, "booking_timestamp"]
    train = df[df["booking_timestamp"] < cutoff_time].copy()
    test = df[df["booking_timestamp"] >= cutoff_time].copy()
    return train, test, cutoff_time


if __name__ == "__main__":
    raw = load_raw()
    feat = build_features(raw)
    train, test, cutoff = chronological_split(feat)
    print(f"Total rows: {len(feat)} | train: {len(train)} | test: {len(test)} | cutoff: {cutoff}")
    print(feat[FEATURE_COLUMNS_NUMERIC].isna().sum().sum(), "NaNs remaining in numeric features")
