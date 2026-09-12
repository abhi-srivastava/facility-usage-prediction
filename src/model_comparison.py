"""
Two additional rigor checks beyond the single train/test split used by the
main pipeline:

1. Model comparison: Random Forest (used in production) vs. Histogram
   Gradient Boosting (scikit-learn's built-in, no extra dependency), on the
   same chronological split, for the facility model.
2. Rolling-origin (walk-forward) cross-validation: instead of trusting one
   train/test cutoff, slide the cutoff across several later points in time
   and report mean +/- std of facility accuracy across folds. This checks
   whether the headline numbers in outputs/metrics.json are representative
   or a lucky/unlucky single split.

Run: python -m src.model_comparison
Output: outputs/model_comparison.json
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.features import load_raw, build_features, FEATURE_COLUMNS_NUMERIC, FEATURE_COLUMNS_CATEGORICAL
from src.models import DesignMatrix, make_facility_model, RANDOM_STATE

OUT_PATH = "outputs/model_comparison.json"


def fit_eval_facility(model_ctor, train, test):
    dm = DesignMatrix()
    X_train = dm.fit_transform(train)
    X_test = dm.transform(test)
    model = model_ctor().fit(X_train, train["facility"])
    pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(test["facility"], pred)),
        "macro_f1": float(f1_score(test["facility"], pred, average="macro")),
    }


def make_hgb_facility_model():
    return HistGradientBoostingClassifier(max_depth=8, max_iter=300, random_state=RANDOM_STATE)


def model_family_comparison(feat: pd.DataFrame) -> dict:
    from src.features import chronological_split
    train, test, cutoff = chronological_split(feat, test_frac=0.18)
    rf_result = fit_eval_facility(make_facility_model, train, test)
    hgb_result = fit_eval_facility(make_hgb_facility_model, train, test)
    return {
        "split_cutoff": str(cutoff), "train_rows": len(train), "test_rows": len(test),
        "random_forest": rf_result, "hist_gradient_boosting": hgb_result,
    }


def rolling_origin_cv(feat: pd.DataFrame, n_folds=4, test_frac_per_fold=0.10) -> dict:
    """Walk the test window forward across the timeline. Fold k trains on
    everything before its window and tests on that window only -- each fold's
    features are still computed leakage-safely from the FULL dataset (every
    row's features only ever look at bookings strictly before it), so this is
    just evaluating at several different chronological cutoffs."""
    feat = feat.sort_values("booking_timestamp").reset_index(drop=True)
    n = len(feat)
    fold_size = int(n * test_frac_per_fold)
    # reserve the earliest ~ (1 - n_folds*test_frac_per_fold) as pure warm-up training data
    warmup = n - n_folds * fold_size

    fold_results = []
    for k in range(n_folds):
        test_start = warmup + k * fold_size
        test_end = test_start + fold_size
        train = feat.iloc[:test_start]
        test = feat.iloc[test_start:test_end]
        if len(train) < 500 or len(test) < 50:
            continue
        result = fit_eval_facility(make_facility_model, train, test)
        fold_results.append({
            "fold": k, "train_rows": len(train), "test_rows": len(test),
            "test_window_start": str(test["booking_timestamp"].min()),
            "test_window_end": str(test["booking_timestamp"].max()),
            **result,
        })

    accs = [f["accuracy"] for f in fold_results]
    return {
        "folds": fold_results,
        "facility_accuracy_mean": float(np.mean(accs)),
        "facility_accuracy_std": float(np.std(accs)),
    }


def main():
    raw = load_raw()
    feat = build_features(raw)

    comparison = model_family_comparison(feat)
    cv = rolling_origin_cv(feat)

    result = {"model_family_comparison": comparison, "rolling_origin_cv": cv}
    with open(OUT_PATH, "w") as fh:
        json.dump(result, fh, indent=2)

    print("Model family comparison (facility prediction, single chronological split):")
    print(f"  Random Forest:            acc={comparison['random_forest']['accuracy']:.3f}  "
          f"macro-F1={comparison['random_forest']['macro_f1']:.3f}")
    print(f"  Hist Gradient Boosting:   acc={comparison['hist_gradient_boosting']['accuracy']:.3f}  "
          f"macro-F1={comparison['hist_gradient_boosting']['macro_f1']:.3f}")
    print(f"\nRolling-origin CV over {len(cv['folds'])} folds: "
          f"facility accuracy {cv['facility_accuracy_mean']:.3f} +/- {cv['facility_accuracy_std']:.3f}")
    for f in cv["folds"]:
        print(f"  fold {f['fold']}: {f['test_window_start']} -> {f['test_window_end']}  acc={f['accuracy']:.3f}")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
