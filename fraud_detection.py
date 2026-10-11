"""
Credit card fraud detection on imbalanced data.
Uses feature engineering, SMOTE (training only), scaling, multiple classifiers,
stratified CV, and hyperparameter tuning for XGBoost.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
# SMOTE target ratio = minority_count / majority_count after resampling (not full 1:1; faster, still strong uplift)
SMOTE_RATIO = 0.12
DATA_PATH = Path(__file__).resolve().parent / "data" / "creditcard.csv"
N_JOBS = -1


def load_and_engineer(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    y = df["Class"].values.astype(np.int64)
    X = df.drop(columns=["Class"])

    X = X.copy()
    X["Amount_log"] = np.log1p(X["Amount"].astype(np.float64))
    X["Time_hour"] = (X["Time"] % 86400) / 3600.0

    feature_cols = [c for c in X.columns if c not in ("Time", "Amount")]
    X = X[feature_cols]

    return X, y


def scale_train_test(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, RobustScaler]:
    scaler = RobustScaler()
    Xt = scaler.fit_transform(X_train)
    Xv = scaler.transform(X_test)
    return Xt, Xv, scaler


def smote_train(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=SMOTE_RATIO)
    return smote.fit_resample(X_train, y_train)


def evaluate_model(name: str, y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_true, y_score),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def cross_val_roc_auc(estimator, X_train: np.ndarray, y_train: np.ndarray, n_splits: int = 3) -> float:
    """Each fold: scale -> SMOTE on train fold only -> fit -> proba on validation fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores: list[float] = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_va = X_train[train_idx], X_train[val_idx]
        y_tr, y_va = y_train[train_idx], y_train[val_idx]
        pipe = ImbPipeline(
            [
                ("scaler", RobustScaler()),
                ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=SMOTE_RATIO)),
                ("clf", estimator),
            ]
        )
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_va)[:, 1]
        scores.append(roc_auc_score(y_va, proba))
    return float(np.mean(scores))


def main(quick: bool = False):
    if not DATA_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATA_PATH}")

    if quick:
        print("Mode: --quick (skip CV and hyperparameter search; test-set metrics only)\n")

    print("Loading data and engineering features...")
    X_df, y = load_and_engineer(DATA_PATH)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    X_train_raw = X_train_df.to_numpy(dtype=np.float64)
    X_test_raw = X_test_df.to_numpy(dtype=np.float64)

    X_train, X_test, _ = scale_train_test(X_train_raw, X_test_raw)
    print(
        f"Train shape: {X_train.shape}, Test shape: {X_test.shape}, "
        f"fraud rate train: {y_train.mean():.4f}"
    )

    X_res, y_res = smote_train(X_train, y_train)
    print(f"After SMOTE — train shape: {X_res.shape}, positive rate: {y_res.mean():.4f}\n")

    results: list[dict] = []

    lr = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    lr.fit(X_res, y_res)
    lr_score = lr.predict_proba(X_test)[:, 1]
    results.append(evaluate_model("Logistic Regression", y_test, lr_score))
    print("=== Logistic Regression (test) ===")
    y_pred_lr = (lr_score >= 0.5).astype(int)
    print(classification_report(y_test, y_pred_lr, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_lr), "\n")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=16,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    rf.fit(X_res, y_res)
    rf_score = rf.predict_proba(X_test)[:, 1]
    results.append(evaluate_model("Random Forest", y_test, rf_score))
    print("=== Random Forest (test) ===")
    y_pred_rf = (rf_score >= 0.5).astype(int)
    print(classification_report(y_test, y_pred_rf, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_rf), "\n")

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        eval_metric="logloss",
        tree_method="hist",
    )
    xgb.fit(X_res, y_res)
    xgb_score = xgb.predict_proba(X_test)[:, 1]
    results.append(evaluate_model("XGBoost (default HP)", y_test, xgb_score))

    search = None
    best_pipe = None

    if not quick:
        print("--- Stratified CV mean ROC-AUC (scale + SMOTE inside each fold; 3-fold) ---")
        # Stratified subsample speeds CV while keeping rare-class representation
        n_cv = min(60_000, X_train_raw.shape[0])
        if X_train_raw.shape[0] > n_cv:
            X_cv, _, y_cv, _ = train_test_split(
                X_train_raw, y_train, train_size=n_cv, stratify=y_train, random_state=RANDOM_STATE
            )
            print(f"  CV subsample size: {n_cv} (of {X_train_raw.shape[0]} training rows)")
        else:
            X_cv, y_cv = X_train_raw, y_train

        lr_cv = cross_val_roc_auc(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs"),
            X_cv,
            y_cv,
        )
        rf_cv = cross_val_roc_auc(
            RandomForestClassifier(
                n_estimators=80,
                max_depth=14,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
            ),
            X_cv,
            y_cv,
        )
        xgb_cv = cross_val_roc_auc(
            XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
                eval_metric="logloss",
                tree_method="hist",
            ),
            X_cv,
            y_cv,
        )
        print(f"  Logistic Regression CV ROC-AUC: {lr_cv:.4f}")
        print(f"  Random Forest CV ROC-AUC:      {rf_cv:.4f}")
        print(f"  XGBoost CV ROC-AUC:            {xgb_cv:.4f}\n")

        print("Running RandomizedSearchCV for XGBoost (SMOTE inside CV; may take several minutes)...")
        xgb_pipe = ImbPipeline(
            [
                ("scaler", RobustScaler()),
                ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=SMOTE_RATIO)),
                (
                    "clf",
                    XGBClassifier(
                        random_state=RANDOM_STATE,
                        n_jobs=N_JOBS,
                        eval_metric="logloss",
                        tree_method="hist",
                    ),
                ),
            ]
        )
        param_dist = {
            "clf__max_depth": [4, 6, 8, 10],
            "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "clf__n_estimators": [200, 400, 600],
            "clf__subsample": [0.7, 0.85, 1.0],
            "clf__colsample_bytree": [0.7, 0.85, 1.0],
            "clf__min_child_weight": [1, 3, 5],
            "clf__gamma": [0, 0.1, 0.2],
        }
        search = RandomizedSearchCV(
            estimator=xgb_pipe,
            param_distributions=param_dist,
            n_iter=10,
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=1,
        )
        search.fit(X_cv, y_cv)
        best_pipe = search.best_estimator_
        xgb_tuned_score = best_pipe.predict_proba(X_test_raw)[:, 1]
        results.append(evaluate_model("XGBoost (tuned)", y_test, xgb_tuned_score))

        print(f"\nBest XGBoost params: {search.best_params_}")
        print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
        print("\n=== XGBoost tuned (test) ===")
        y_pred_xgb = (xgb_tuned_score >= 0.5).astype(int)
        print(classification_report(y_test, y_pred_xgb, digits=4))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_xgb))

    summary = pd.DataFrame(results)
    print("\n========== Test set summary ==========")
    print(summary.to_string(index=False))
    best_row = summary.loc[summary["roc_auc"].idxmax()]
    print(f"\nBest test ROC-AUC: {best_row['model']} ({best_row['roc_auc']:.4f})")

    if best_pipe is not None:
        X_sub, X_val, y_sub, y_val = train_test_split(
            X_train_raw, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
        )
        val_pipe = search.best_estimator_
        val_pipe.fit(X_sub, y_sub)
        val_proba = val_pipe.predict_proba(X_val)[:, 1]
        best_f1_th = 0.5
        best_f1 = 0.0
        for th in np.linspace(0.05, 0.95, 91):
            pred = (val_proba >= th).astype(int)
            f1 = f1_score(y_val, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_f1_th = f1, th
        best_pipe.fit(X_train_raw, y_train)
        test_proba = best_pipe.predict_proba(X_test_raw)[:, 1]
        y_pred_th = (test_proba >= best_f1_th).astype(int)
        print(f"\n--- XGBoost tuned @ F1-optimal threshold ({best_f1_th:.3f} on val) ---")
        print(
            f"ROC-AUC: {roc_auc_score(y_test, test_proba):.4f} | "
            f"Precision: {precision_score(y_test, y_pred_th, zero_division=0):.4f} | "
            f"Recall: {recall_score(y_test, y_pred_th, zero_division=0):.4f} | "
            f"F1: {f1_score(y_test, y_pred_th, zero_division=0):.4f}"
        )
    elif quick:
        print("\n(Run without --quick for CV scores, XGBoost tuning, and threshold sweep.)")


def _parse_args():
    p = argparse.ArgumentParser(description="Credit card fraud detection training")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Only fit LR/RF/XGB on SMOTE data and evaluate on test (minutes, not hours).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(quick=args.quick)
