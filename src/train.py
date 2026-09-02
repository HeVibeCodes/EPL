"""
Trains three model heads on the feature table built by features.py:

  1. Match outcome (Home win / Draw / Away win)  -- multiclass, calibrated
  2. Total goals over/under 2.5                   -- binary
  3. Total cards over/under 3.5                   -- binary (referee-driven)

Two things changed from the first version of this script, based on what the
initial numbers showed:

  1. TEST SET: testing against the current in-progress season (sometimes as
     few as 20 matches played) makes every metric noisy and close to
     meaningless -- a handful of unlucky results can swing "accuracy" by 10+
     points either way. `chronological_split()` now finds the most recent
     COMPLETE season (>= min_test_matches games, with enough matches left
     over to actually train on) and tests on that instead.

  2. MODEL CHOICE + BALANCING: rather than hard-coding GradientBoosting,
     each target now runs a small model bake-off (logistic regression,
     random forest, gradient boosting, hist-gradient-boosting) scored with
     time-series cross-validation, and the best performer is the one
     actually fit and evaluated. Every candidate -- including the two
     boosting models, which have no `class_weight` parameter -- is trained
     with class-balanced SAMPLE weights instead, so "predict the majority
     class every time" stops being a free win. For the two binary markets,
     the 0.5 probability cutoff is also replaced with a threshold tuned on
     a held-out slice of training data to maximize sensitivity+specificity
     (Youden's J), instead of assuming 0.5 is the right cutoff.

Sensitivity (recall on the "yes" side) and specificity (recall on the "no"
side) are reported alongside accuracy/AUC for every target, since a single
accuracy number hides whether a model is just calling everything "home win"
/ "under" and coasting on class imbalance.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import features
from features import build_feature_table

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Data split
# --------------------------------------------------------------------------- #
def chronological_split(df: pd.DataFrame, min_test_matches: int = 300, min_train_matches: int = 300):
    """
    Test on the most recent season with at least `min_test_matches` games
    (i.e. a season that's actually finished) that STILL leaves at least
    `min_train_matches` earlier games to train on. Falls back to an 80/20
    split by date if no season qualifies (e.g. only one season on file).
    """
    seasons_desc = sorted(df["season"].unique(), reverse=True)
    for season in seasons_desc:
        n = (df["season"] == season).sum()
        if n < min_test_matches:
            continue
        test = df[df["season"] == season]
        train = df[df["Date"] < test["Date"].min()]
        if len(train) >= min_train_matches:
            return train, test

    df = df.sort_values("Date")
    cutoff = int(len(df) * 0.8)
    return df.iloc[:cutoff], df.iloc[cutoff:]


def make_matrix(df: pd.DataFrame, feature_cols: list[str], fill_values: pd.Series | None = None):
    X = df[feature_cols].copy()
    fill_values = fill_values if fill_values is not None else X.median(numeric_only=True)
    X = X.fillna(fill_values)
    return X, fill_values


# --------------------------------------------------------------------------- #
# Model bake-off
# --------------------------------------------------------------------------- #
def candidate_models():
    """
    class_weight isn't set here -- balancing is applied uniformly via
    sample_weight at fit time instead (see fit_balanced()), so every
    candidate gets the same treatment regardless of whether it has a
    class_weight param (GradientBoosting/HistGradientBoosting don't).
    """
    return {
        "logistic_regression": LogisticRegression(max_iter=3000),
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=7, min_samples_leaf=5, random_state=42, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05, random_state=42
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.06, max_iter=300, random_state=42
        ),
    }


def fit_balanced(pipe: Pipeline, X, y):
    """Fit a Pipeline([('scale', ...), ('clf', ...)]) with class-balanced
    sample weights, routed to whichever estimator sits at the 'clf' step
    (works whether that's a raw classifier or a CalibratedClassifierCV
    wrapping one -- both accept sample_weight)."""
    weights = compute_sample_weight("balanced", y)
    pipe.fit(X, y, clf__sample_weight=weights)
    return pipe


def select_best_model(X, y, scoring: str, n_splits: int = 5, label: str = ""):
    """Manual time-series CV bake-off (not cross_val_score, so we can apply
    balanced sample weights correctly within each fold). Returns
    (name, unfitted estimator)."""
    n_splits = max(2, min(n_splits, len(X) // 40))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    scores = {}
    for name, clf in candidate_models().items():
        fold_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            if y_tr.nunique() < 2 or y_val.nunique() < 1:
                continue
            try:
                pipe = Pipeline([("scale", StandardScaler()), ("clf", clf)])
                fit_balanced(pipe, X_tr, y_tr)
                if scoring == "accuracy":
                    fold_scores.append(accuracy_score(y_val, pipe.predict(X_val)))
                elif scoring == "roc_auc" and y_val.nunique() > 1:
                    fold_scores.append(roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1]))
            except Exception as exc:
                print(f"    {name}: fold failed ({exc})")
        if fold_scores:
            scores[name] = float(np.mean(fold_scores))

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    print(f"  [{label}] model comparison ({scoring}, {n_splits}-fold time-series CV, balanced weights):")
    for name, score in ranked:
        print(f"    {name:24s} {score:.3f}")

    if not ranked:
        raise RuntimeError(
            f"[{label}] every candidate model failed cross-validation -- "
            f"likely too little training data ({len(X)} rows) for {n_splits} folds."
        )

    best_name = ranked[0][0]
    print(f"  [{label}] selected: {best_name}")
    return best_name, candidate_models()[best_name]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def binary_sens_spec(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),  # recall on the "yes" class
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),  # recall on the "no" class
    }


def multiclass_sens_spec(y_true, y_pred, classes=(0, 1, 2), names=("Home", "Draw", "Away")) -> dict:
    out = {}
    for c, name in zip(classes, names):
        out[name] = binary_sens_spec((y_true == c).astype(int), (y_pred == c).astype(int))
    return out


def find_balanced_threshold(y_true, proba) -> float:
    """
    The default 0.5 cutoff assumes a roughly balanced, well-separated
    classifier -- neither is guaranteed here. Instead pick the threshold
    that maximizes sensitivity + specificity (Youden's J), so the model
    can't just default to calling everything "over" and collect a cheap
    accuracy score.
    """
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        pred = (proba > t).astype(int)
        m = binary_sens_spec(y_true, pred)
        score = m["sensitivity"] + m["specificity"]
        if score > best_score:
            best_score, best_t = score, t
    return float(best_t)


# --------------------------------------------------------------------------- #
# Train each head
# --------------------------------------------------------------------------- #
def train_1x2(train, test, feature_cols):
    X_train, fill_values = make_matrix(train, feature_cols)
    y_train = train["Target_1X2"]
    X_test, _ = make_matrix(test, feature_cols, fill_values)
    y_test = test["Target_1X2"]

    best_name, base = select_best_model(X_train, y_train, scoring="accuracy", label="1X2")
    pipe = Pipeline([("scale", StandardScaler()), ("clf", CalibratedClassifierCV(base, cv=3, method="isotonic"))])
    fit_balanced(pipe, X_train, y_train)

    proba = pipe.predict_proba(X_test)
    pred = pipe.predict(X_test)
    metrics = {
        "model": best_name,
        "accuracy": accuracy_score(y_test, pred),
        "log_loss": log_loss(y_test, proba, labels=[0, 1, 2]),
        "per_class": multiclass_sens_spec(y_test.to_numpy(), pred),
    }
    joblib.dump(pipe, os.path.join(MODEL_DIR, "model_1x2.joblib"))
    joblib.dump(fill_values, os.path.join(MODEL_DIR, "model_1x2_fillvalues.joblib"))
    return metrics


def train_binary(train, test, feature_cols, target_col, model_name, label):
    X_train, fill_values = make_matrix(train, feature_cols)
    y_train = train[target_col]
    X_test, _ = make_matrix(test, feature_cols, fill_values)
    y_test = test[target_col]

    best_name, base = select_best_model(X_train, y_train, scoring="roc_auc", label=label)
    pipe = Pipeline([("scale", StandardScaler()), ("clf", CalibratedClassifierCV(base, cv=3, method="sigmoid"))])

    # tune the decision threshold on a held-out TAIL of the training data
    # (never on test) so the final cutoff isn't just a blind 0.5
    split = int(len(X_train) * 0.85)
    X_fit, X_val = X_train.iloc[:split], X_train.iloc[split:]
    y_fit, y_val = y_train.iloc[:split], y_train.iloc[split:]
    tuning_pipe = Pipeline([("scale", StandardScaler()), ("clf", CalibratedClassifierCV(base, cv=3, method="sigmoid"))])
    threshold = 0.5
    if y_fit.nunique() > 1 and y_val.nunique() > 1:
        fit_balanced(tuning_pipe, X_fit, y_fit)
        val_proba = tuning_pipe.predict_proba(X_val)[:, 1]
        threshold = find_balanced_threshold(y_val.to_numpy(), val_proba)

    fit_balanced(pipe, X_train, y_train)  # final model fit on ALL training data

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba > threshold).astype(int)
    metrics = {
        "model": best_name,
        "decision_threshold": round(threshold, 2),
        "accuracy": accuracy_score(y_test, pred),
        "auc": roc_auc_score(y_test, proba) if y_test.nunique() > 1 else float("nan"),
        "brier": brier_score_loss(y_test, proba),
        **binary_sens_spec(y_test.to_numpy(), pred),
    }
    joblib.dump(pipe, os.path.join(MODEL_DIR, f"{model_name}.joblib"))
    joblib.dump(fill_values, os.path.join(MODEL_DIR, f"{model_name}_fillvalues.joblib"))
    joblib.dump(threshold, os.path.join(MODEL_DIR, f"{model_name}_threshold.joblib"))
    return metrics


def baseline_class_split(train):
    """What you'd get by always predicting the historical majority class."""
    return train["Target_1X2"].value_counts(normalize=True).to_dict()


def main():
    df = build_feature_table()
    train, test = chronological_split(df, min_test_matches=300)
    print(f"Train: {len(train)} matches ({train['season'].min()}-{train['season'].max()})")
    print(f"Test:  {len(test)} matches (season {sorted(test['season'].unique())})\n")

    feature_cols = features.FEATURE_COLUMNS
    results = {}
    results["1X2"] = train_1x2(train, test, feature_cols)
    print()
    results["OverUnder2.5"] = train_binary(train, test, feature_cols, "Target_Over2.5", "model_over25", "Over 2.5 goals")
    print()
    results["CardsOver3.5"] = train_binary(train, test, feature_cols, "Target_CardsOver3.5", "model_cards35", "Over 3.5 cards")
    print()
    results["baseline_class_split"] = baseline_class_split(train)

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(json.dumps(results, indent=2, default=str))
    print(f"\nModels saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
