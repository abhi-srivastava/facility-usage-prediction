"""
Mechanical proof that the feature pipeline is leakage-safe.

Rather than just asserting this in prose, we perturb the dataset and check
that a row's computed features are invariant to changes we make in the
future (after that row's booking_timestamp) and DO change when we perturb
the past (a sanity check that the test isn't vacuously true because
features are constants / broken).

Runnable standalone (no pytest required): `python tests/test_leakage.py`
Also discoverable by pytest if installed: `pytest tests/`
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import load_raw, build_features, FEATURE_COLUMNS_NUMERIC, FEATURE_COLUMNS_CATEGORICAL

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "bookings.csv")


def _perturb_future_rows(raw: pd.DataFrame, cutoff_ts: pd.Timestamp, rng: np.random.Generator) -> pd.DataFrame:
    """Rewrite facility + usage_timestamp for every row at/after cutoff_ts, to
    simulate 'the future turned out completely differently'. booking_timestamp
    (the sort key / leakage boundary) is left untouched."""
    df = raw.copy()
    future_mask = df["booking_timestamp"] >= cutoff_ts
    facilities = df["facility"].unique()
    df.loc[future_mask, "facility"] = rng.choice(facilities, size=future_mask.sum())
    jitter_hours = rng.uniform(-500, 500, size=future_mask.sum())
    df.loc[future_mask, "usage_timestamp"] = df.loc[future_mask, "usage_timestamp"] + pd.to_timedelta(jitter_hours, unit="h")
    return df


def _perturb_past_rows(raw: pd.DataFrame, cutoff_ts: pd.Timestamp, rng: np.random.Generator) -> pd.DataFrame:
    """The mirror-image perturbation: rewrite the PAST instead of the future."""
    df = raw.copy()
    past_mask = df["booking_timestamp"] < cutoff_ts
    facilities = df["facility"].unique()
    df.loc[past_mask, "facility"] = rng.choice(facilities, size=past_mask.sum())
    return df


def _pick_probe_booking_id(raw: pd.DataFrame, baseline_feat: pd.DataFrame) -> str:
    """A booking roughly in the middle of a resident's history, so both
    'this resident's later bookings' and 'other residents' later bookings'
    can be perturbed against it."""
    counts = raw["resident_id"].value_counts()
    resident = counts[counts >= 6].index[0]
    resident_rows = baseline_feat[baseline_feat["resident_id"] == resident].sort_values("booking_timestamp")
    return resident_rows.iloc[len(resident_rows) // 2]["booking_id"]


def test_features_unaffected_by_future_perturbation():
    rng = np.random.default_rng(7)
    raw = load_raw(DATA_PATH)

    baseline_feat = build_features(raw)
    probe_booking_id = _pick_probe_booking_id(raw, baseline_feat)
    probe_ts = baseline_feat.loc[baseline_feat["booking_id"] == probe_booking_id, "booking_timestamp"].iloc[0]

    perturbed_raw = _perturb_future_rows(raw, probe_ts + pd.Timedelta(seconds=1), rng)
    perturbed_feat = build_features(perturbed_raw)

    cols = FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL
    before = baseline_feat.loc[baseline_feat["booking_id"] == probe_booking_id, cols].iloc[0]
    after = perturbed_feat.loc[perturbed_feat["booking_id"] == probe_booking_id, cols].iloc[0]

    mismatches = [c for c in cols if not _values_equal(before[c], after[c])]
    assert not mismatches, (
        f"Leakage detected: features {mismatches} for booking {probe_booking_id} changed "
        f"after perturbing only rows at/after its own booking_timestamp ({probe_ts}). "
        f"before={before[mismatches].to_dict()} after={after[mismatches].to_dict()}"
    )


def test_features_do_change_with_past_perturbation():
    """Sanity check: if we instead scramble the PAST, the same row's history
    features SHOULD change -- otherwise the previous test would be trivially
    true because the features are constants or broken."""
    rng = np.random.default_rng(11)
    raw = load_raw(DATA_PATH)
    baseline_feat = build_features(raw)
    probe_booking_id = _pick_probe_booking_id(raw, baseline_feat)

    probe_ts = baseline_feat.loc[baseline_feat["booking_id"] == probe_booking_id, "booking_timestamp"].iloc[0]
    n_prior = baseline_feat.loc[baseline_feat["booking_id"] == probe_booking_id, "n_prior_bookings"].iloc[0]
    assert n_prior > 0, "probe row must have prior history for this check to be meaningful"

    perturbed_raw = _perturb_past_rows(raw, probe_ts, rng)
    perturbed_feat = build_features(perturbed_raw)

    cols = [f"pref_{f}" for f in ["Gym", "Swimming Pool", "Badminton Court", "Tennis Court"]]
    before = baseline_feat.loc[baseline_feat["booking_id"] == probe_booking_id, cols].iloc[0]
    after = perturbed_feat.loc[perturbed_feat["booking_id"] == probe_booking_id, cols].iloc[0]

    changed = any(not _values_equal(before[c], after[c]) for c in cols)
    assert changed, (
        "Expected resident facility-preference features to change when the PAST is scrambled "
        "-- if they don't, the future-leakage test above may be vacuous."
    )


def test_no_nan_in_features():
    raw = load_raw(DATA_PATH)
    feat = build_features(raw)
    n_nan = feat[FEATURE_COLUMNS_NUMERIC].isna().sum().sum()
    assert n_nan == 0, f"Found {n_nan} NaN values in numeric features (cold-start rows should be imputed)"


def _values_equal(a, b, tol=1e-9):
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    if isinstance(a, (int, float, np.floating, np.integer)) and isinstance(b, (int, float, np.floating, np.integer)):
        return abs(float(a) - float(b)) <= tol
    return a == b


def _run_all():
    print("test_no_nan_in_features ...", end=" ")
    test_no_nan_in_features()
    print("PASS")

    print("test_features_unaffected_by_future_perturbation ...", end=" ")
    test_features_unaffected_by_future_perturbation()
    print("PASS")

    print("test_features_do_change_with_past_perturbation ...", end=" ")
    test_features_do_change_with_past_perturbation()
    print("PASS")

    print("\nAll leakage-safety tests passed.")


if __name__ == "__main__":
    _run_all()
