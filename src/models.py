"""Model definitions and a shared design-matrix builder for the four prediction heads."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

from src.features import FEATURE_COLUMNS_NUMERIC, FEATURE_COLUMNS_CATEGORICAL

RANDOM_STATE = 42


class DesignMatrix:
    """One-hot encodes the small categorical feature set (fit on train only) and
    concatenates it with the numeric features into a single float matrix.

    `extra_categorical` lets the day/hour/lead-time models additionally
    condition on a "facility_context" column: the ACTUAL facility at train
    time (a known fact about that historical booking), and the FACILITY
    MODEL's PREDICTED facility at evaluation/inference time (since the real
    future facility isn't known yet). This cascaded design captures that
    usage day/hour and lead time depend heavily on which facility is being
    booked (e.g. Gym slots vs. Clubhouse slots)."""

    def __init__(self, extra_categorical=None):
        self.categorical_cols = FEATURE_COLUMNS_CATEGORICAL + (extra_categorical or [])
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit(self, df: pd.DataFrame):
        self.encoder.fit(df[self.categorical_cols])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cat = self.encoder.transform(df[self.categorical_cols])
        num = df[FEATURE_COLUMNS_NUMERIC].to_numpy(dtype=float)
        return np.hstack([num, cat])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


def make_facility_model():
    return RandomForestClassifier(
        n_estimators=200, max_depth=11, min_samples_leaf=5,
        class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1,
    )


def make_dow_model():
    return RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1,
    )


def make_hour_model():
    return RandomForestClassifier(
        n_estimators=200, max_depth=11, min_samples_leaf=4,
        random_state=RANDOM_STATE, n_jobs=-1,
    )


def make_leadtime_model():
    # regression on log1p(lead_time_hours) to tame the right-skewed distribution
    return RandomForestRegressor(
        n_estimators=200, max_depth=11, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
