"""TSTR and fidelity evaluation for real SDG baselines.

Computes:
  - Real-data oracle  : train on real, test on real (upper bound)
  - TSTR              : train on synthetic, test on real
  - Wasserstein dist  : marginal distributional fidelity per numeric column
  - Correlation delta : Frobenius norm of correlation matrix difference
  - Utility retention : TSTR F1 / oracle F1
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class EvalResult:
    dataset: str
    synthesizer: str
    oracle_f1: float
    oracle_auc: float
    tstr_f1: float
    tstr_auc: float
    utility_retention_f1: float
    utility_retention_auc: float
    wasserstein_mean: float
    correlation_delta: float
    n_synthetic: int
    n_real_train: int
    n_real_test: int

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "synthesizer": self.synthesizer,
            "oracle_f1": round(self.oracle_f1, 4),
            "oracle_auc": round(self.oracle_auc, 4),
            "tstr_f1": round(self.tstr_f1, 4),
            "tstr_auc": round(self.tstr_auc, 4),
            "utility_retention_f1": round(self.utility_retention_f1, 4),
            "utility_retention_auc": round(self.utility_retention_auc, 4),
            "wasserstein_mean": round(self.wasserstein_mean, 4),
            "correlation_delta": round(self.correlation_delta, 4),
            "n_synthetic": self.n_synthetic,
            "n_real_train": self.n_real_train,
            "n_real_test": self.n_real_test,
        }


def _encode(df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray]:
    """One-hot encode categoricals and return (X, y) numpy arrays."""
    from sklearn.preprocessing import LabelEncoder
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].values.astype(int)
    # One-hot encode object/string columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    return X.values.astype(float), y


def _fill_missing_cols(
    X_synth: pd.DataFrame,
    X_real: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """After one-hot encoding, align synthetic and real columns."""
    all_cols = sorted(set(X_synth.columns) | set(X_real.columns))
    for col in all_cols:
        if col not in X_synth.columns:
            X_synth[col] = 0
        if col not in X_real.columns:
            X_real[col] = 0
    X_synth = X_synth[all_cols]
    X_real = X_real[all_cols]
    return X_synth.values.astype(float), X_real.values.astype(float)


def _train_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float]:
    """Train LogisticRegression, return (F1, AUC)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=500, random_state=42, C=1.0)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.5
    return float(f1), float(auc)


def _wasserstein_mean(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
    """Mean Wasserstein-1 distance across numeric columns (normalized by std)."""
    from scipy.stats import wasserstein_distance
    num_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    distances = []
    for col in num_cols:
        if col not in synth_df.columns:
            continue
        r = real_df[col].dropna().values
        s = synth_df[col].dropna().values
        std = r.std() or 1.0
        d = wasserstein_distance(r / std, s / std)
        distances.append(d)
    return float(np.mean(distances)) if distances else 0.0


def _correlation_delta(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
    """Frobenius norm of (real_corr - synth_corr) for numeric columns."""
    num_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    common = [c for c in num_cols if c in synth_df.columns]
    if len(common) < 2:
        return 0.0
    r_corr = real_df[common].corr().fillna(0).values
    s_corr = synth_df[common].corr().fillna(0).values
    return float(np.linalg.norm(r_corr - s_corr, "fro"))


def evaluate(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    target_col: str,
    dataset_name: str,
    synthesizer_name: str,
    test_size: float = 0.20,
    seed: int = 42,
) -> EvalResult:
    """Full TSTR + fidelity evaluation."""
    from sklearn.model_selection import train_test_split

    # ── Split real data 80 / 20 ──────────────────────────────────────
    train_real, test_real = train_test_split(
        real_df, test_size=test_size, random_state=seed, stratify=real_df[target_col]
    )

    # ── Encode separately then align columns ─────────────────────────
    # Real training set
    X_real_tr_raw = pd.get_dummies(
        train_real.drop(columns=[target_col]),
        drop_first=True,
    )
    y_real_tr = train_real[target_col].values.astype(int)

    # Real test set
    X_real_te_raw = pd.get_dummies(
        test_real.drop(columns=[target_col]),
        drop_first=True,
    )
    y_real_te = test_real[target_col].values.astype(int)

    # Synthetic
    synth_with_target = synth_df.copy()
    if target_col not in synth_with_target.columns:
        # Fallback: use 0 labels — won't happen with proper synthesis
        synth_with_target[target_col] = 0
    X_synth_raw = pd.get_dummies(
        synth_with_target.drop(columns=[target_col]),
        drop_first=True,
    )
    y_synth = synth_with_target[target_col].values.astype(int)

    # Align real-train ↔ real-test
    X_real_tr, X_real_te = _fill_missing_cols(X_real_tr_raw.copy(), X_real_te_raw.copy())

    # Align synth ↔ real-test
    X_synth_a, X_real_te2 = _fill_missing_cols(X_synth_raw.copy(), X_real_te_raw.copy())

    # ── Oracle: train on real, test on real ──────────────────────────
    oracle_f1, oracle_auc = _train_eval(X_real_tr, y_real_tr, X_real_te, y_real_te)

    # ── TSTR: train on synthetic, test on real ───────────────────────
    tstr_f1, tstr_auc = _train_eval(X_synth_a, y_synth, X_real_te2, y_real_te)

    # ── Fidelity ─────────────────────────────────────────────────────
    wd = _wasserstein_mean(train_real, synth_df)
    cd = _correlation_delta(train_real, synth_df)

    return EvalResult(
        dataset=dataset_name,
        synthesizer=synthesizer_name,
        oracle_f1=oracle_f1,
        oracle_auc=oracle_auc,
        tstr_f1=tstr_f1,
        tstr_auc=tstr_auc,
        utility_retention_f1=tstr_f1 / max(oracle_f1, 1e-9),
        utility_retention_auc=tstr_auc / max(oracle_auc, 1e-9),
        wasserstein_mean=wd,
        correlation_delta=cd,
        n_synthetic=len(synth_df),
        n_real_train=len(train_real),
        n_real_test=len(test_real),
    )
