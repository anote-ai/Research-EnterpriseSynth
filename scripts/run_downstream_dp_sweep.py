#!/usr/bin/env python3
"""Real DP-SGD sweep for downstream_tasks.json's regression and anomaly tasks.

Classification's DP values already go through real DP-SGD training via
results/real_dp_sweep.json (wired into results/epsilon_sweep.json). This
script attempts the same for regression and anomaly detection, which
previously only had measured no-DP baselines with DP values estimated via a
logistic degradation curve (see RESEARCH_STATUS.md).

Status: NEITHER TASK IS WIRED INTO results/downstream_tasks.json.  Both
attempts surfaced genuine problems rather than usable measurements, and
reporting them as "measured" would be misleading:

  - Regression (diabetes-progression, RandomForestRegressor R^2): the DPVAE
    consistently collapses to predicting a near-constant target value,
    giving R^2 = 0.0 across all 5 seeds at every tested epsilon -- including
    epsilon=10 (loosest budget tested), which rules out DP noise as the
    cause. Same class of issue as the DPVAE/DPTVAE instability documented
    elsewhere in RESEARCH_STATUS.md, just surfacing for a continuous target
    this time. Not resolved; not wired in.

  - Anomaly (breast-cancer, IsolationForest recall): recall alone is not a
    sufficient metric here. The DP-SGD-trained synthetic "normal" class
    produces an IsolationForest that flags ~100% of the real test set as
    anomalous (precision = 0.374, exactly the test set's true anomaly base
    rate -- i.e. no genuine discrimination, just "flag everything").
    Recall = 1.0 in this case is trivial, not evidence the model works.
    For contrast, the real ORACLE detector (trained on real normal data,
    already committed pre-this-session) is legitimate: recall = 1.0 *with*
    precision = 0.634, a real, non-trivial detector. Only the DP-SGD
    synthetic version is degenerate. Not wired in.

Uses the same DPVAE architecture as run_opacus_dp_sweep.py (not DPTVAE,
which has an unresolved collapse issue on larger/higher-cardinality
datasets) on the same two sklearn datasets already used as no-DP proxies in
scripts/run_downstream_tasks.py.

Usage (exploratory only -- see Status above before trusting any output):
    python scripts/run_downstream_dp_sweep.py                # both tasks
    python scripts/run_downstream_dp_sweep.py --task regression
    python scripts/run_downstream_dp_sweep.py --epsilon 2.0 --seeds 2
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from privacy_benchmark.config import EPSILON_VALUES
from run_opacus_dp_sweep import _DPTabularVAESynthesizer, _mia_auc, _fidelity_score, _check_deps


def _load_regression_dataset():
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    data = load_diabetes(as_frame=True)
    df = data.frame.copy()
    df.columns = [c.lower() for c in df.columns]
    target = "target"
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    df_train = X_train.copy()
    df_train[target] = y_train.values
    return X_train, X_test, y_train, y_test, df_train, target


def _load_anomaly_dataset():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    # sklearn breast-cancer target: 0=malignant, 1=benign. Anomaly = malignant,
    # matching the convention already used in run_downstream_tasks.py.
    df["anomaly"] = (df["target"] == 0).astype(int)
    df = df.drop(columns=["target"])
    target = "anomaly"
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    df_train = X_train.copy()
    df_train[target] = y_train.values
    return X_train, X_test, y_train, y_test, df_train, target


def _oracle_r2(X_train, X_test, y_train, y_test) -> float:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return max(0.0, float(r2_score(y_test, model.predict(X_test))))


def _oracle_anomaly_recall(X_train, X_test, y_train, y_test) -> float:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import recall_score

    normal_train = X_train[y_train == 0]
    model = Pipeline([
        ("scale", StandardScaler()),
        ("iso", IsolationForest(n_estimators=300, contamination=float(y_train.mean()), random_state=42)),
    ])
    model.fit(normal_train)
    preds = (model.predict(X_test) == -1).astype(int)
    return float(recall_score(y_test, preds, zero_division=0))


def _tstr_r2(synth, n_synthetic: int, X_test, y_test, target_col: str) -> float:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    syn_df = synth.sample(num_rows=n_synthetic)
    y_syn = syn_df[target_col]
    X_syn = syn_df.drop(columns=[target_col])
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_syn, y_syn)
    return max(0.0, float(r2_score(y_test, model.predict(X_test))))


def _tstr_anomaly_recall(synth, n_synthetic: int, X_test, y_test, target_col: str) -> float:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import recall_score

    syn_df = synth.sample(num_rows=n_synthetic)
    y_syn = syn_df[target_col].astype(int)
    X_syn = syn_df.drop(columns=[target_col])
    normal_syn = X_syn[y_syn == 0]
    if len(normal_syn) < 5:
        return 0.0
    model = Pipeline([
        ("scale", StandardScaler()),
        ("iso", IsolationForest(n_estimators=300, contamination=0.1, random_state=42)),
    ])
    model.fit(normal_syn)
    preds = (model.predict(X_test) == -1).astype(int)
    return float(recall_score(y_test, preds, zero_division=0))


TASKS = {
    "regression": {
        "loader": _load_regression_dataset,
        "oracle_fn": _oracle_r2,
        "tstr_fn": _tstr_r2,
        "metric_name": "r2",
    },
    "anomaly": {
        "loader": _load_anomaly_dataset,
        "oracle_fn": _oracle_anomaly_recall,
        "tstr_fn": _tstr_anomaly_recall,
        "metric_name": "recall",
    },
}


def run_task_sweep(
    task_name: str,
    epsilon_values: list[float],
    delta: float,
    epochs: int,
    batch_size: int,
    lr: float,
    max_grad_norm: float,
    device: str,
    n_seeds: int,
) -> list[dict]:
    cfg = TASKS[task_name]
    X_train, X_test, y_train, y_test, df_train, target_col = cfg["loader"]()
    oracle = round(cfg["oracle_fn"](X_train, X_test, y_train, y_test), 4)
    print(f"\n{'='*70}\nTask: {task_name}  oracle {cfg['metric_name']}={oracle}  n_train={len(df_train)}")

    synth = _DPTabularVAESynthesizer(df_train, target_col)
    results = []
    for eps in epsilon_values:
        print(f"\n  eps={eps}  ({n_seeds} seeds):")
        metric_values, mia_values, fidelity_values = [], [], []
        elapsed_total = 0.0
        for seed in range(n_seeds):
            t0 = time.time()
            spent_eps, elapsed = synth.fit(
                eps, delta, epochs, batch_size, lr, max_grad_norm, device, seed=1000 * seed + 7
            )
            metric = round(cfg["tstr_fn"](synth, len(df_train), X_test, y_test, target_col), 4)
            mia = _mia_auc(synth, df_train, X_test, y_test, target_col)
            synth_df = synth.sample(num_rows=len(df_train))
            fidelity = _fidelity_score(df_train, synth_df, list(df_train.columns), target_col)
            print(f"    seed={seed}  {cfg['metric_name']}={metric}  mia={mia}  fidelity={fidelity}")
            metric_values.append(metric)
            mia_values.append(mia)
            fidelity_values.append(fidelity)
            elapsed_total += elapsed

        mean_metric = sum(metric_values) / n_seeds
        std_metric = (sum((v - mean_metric) ** 2 for v in metric_values) / n_seeds) ** 0.5
        mean_mia = sum(mia_values) / n_seeds
        mean_fidelity = sum(fidelity_values) / n_seeds
        print(f"    -> mean {cfg['metric_name']}={mean_metric:.4f} (std={std_metric:.4f})")

        results.append({
            "task": task_name,
            "metric_name": cfg["metric_name"],
            "epsilon": eps,
            "delta": delta,
            "oracle": oracle,
            "n_seeds": n_seeds,
            "metric_values": metric_values,
            "metric_mean": round(mean_metric, 4),
            "metric_std": round(std_metric, 4),
            "retention_mean": round(mean_metric / oracle, 4) if oracle > 0 else 0.0,
            "mia_auc_mean": round(mean_mia, 4),
            "fidelity_score_mean": round(mean_fidelity, 4),
            "elapsed_sec": round(elapsed_total, 1),
            "data_source": f"measured — real DP-SGD training, mean over {n_seeds} seeds",
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Real DP-SGD sweep for regression/anomaly tasks")
    parser.add_argument("--task", default="all", choices=list(TASKS) + ["all"])
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    _check_deps()

    tasks = list(TASKS) if args.task == "all" else [args.task]
    eps_values = [args.epsilon] if args.epsilon else EPSILON_VALUES

    all_results: list[dict] = []
    run_context = contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext()
    with run_context:
        for task in tasks:
            all_results.extend(
                run_task_sweep(
                    task, eps_values, args.delta, args.epochs, args.batch_size,
                    args.lr, args.max_grad_norm, args.device, args.seeds,
                )
            )

    if args.json:
        print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
