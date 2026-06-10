"""Statistical rigor utilities for privacy-utility-fidelity benchmarking.

Implements:
  - Bootstrap confidence intervals on utility metrics
  - DP training variance analysis across seeds
  - Significance testing (Wilcoxon signed-rank / paired t-test) with Bonferroni
  - Privacy accountant epsilon verification
  - Dataset-type stratification reporting
"""
from __future__ import annotations

import math
import random
from typing import Callable, Sequence


# ---------------------------------------------------------------------------
# 1. Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: Sequence[float],
    metric_fn: Callable[[Sequence[float]], float] = lambda v: sum(v) / len(v),
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute a bootstrap confidence interval for *metric_fn* applied to *values*.

    Returns (mean, ci_lower, ci_upper) as a 3-tuple.
    """
    if not values:
        raise ValueError("values must not be empty")

    rng = random.Random(seed)
    n = len(values)
    stats: list[float] = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(n)]  # type: ignore[arg-type]
        stats.append(metric_fn(sample))

    stats.sort()
    alpha = 1.0 - confidence
    lo_idx = int(math.floor(alpha / 2 * n_bootstrap))
    hi_idx = int(math.ceil((1.0 - alpha / 2) * n_bootstrap)) - 1
    ci_lower = stats[max(0, lo_idx)]
    ci_upper = stats[min(n_bootstrap - 1, hi_idx)]
    mean = metric_fn(list(values))
    return mean, ci_lower, ci_upper


def bootstrap_tstr_ci(
    tstr_scores: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap CI wrapper for TSTR scores.  Returns a dict with keys
    ``mean``, ``ci_lower``, ``ci_upper``, and ``ci_half_width``."""
    mean, lo, hi = bootstrap_ci(
        tstr_scores,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    return {
        "mean": mean,
        "ci_lower": lo,
        "ci_upper": hi,
        "ci_half_width": (hi - lo) / 2.0,
    }


# ---------------------------------------------------------------------------
# 2. DP Training Variance (multi-seed)
# ---------------------------------------------------------------------------

def dp_training_variance(
    seed_results: Sequence[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Summarise per-metric mean ± std across multiple DP training seeds.

    *seed_results* is a list of dicts, each mapping metric name → value for
    one seed run.  Returns a dict mapping metric name → {mean, std, min, max}.
    """
    if not seed_results:
        raise ValueError("seed_results must not be empty")

    metric_names = list(seed_results[0].keys())
    summary: dict[str, dict[str, float]] = {}

    for metric in metric_names:
        vals = [float(r[metric]) for r in seed_results]
        n = len(vals)
        mean = sum(vals) / n
        variance = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0.0
        summary[metric] = {
            "mean": mean,
            "std": math.sqrt(variance),
            "min": min(vals),
            "max": max(vals),
            "n_seeds": float(n),
        }

    return summary


def epsilon_variance_profile(
    epsilon_to_seed_results: dict[float, Sequence[dict[str, float]]],
    metric: str = "tstr_score",
) -> list[dict[str, float]]:
    """Return a list of {epsilon, mean, std} dicts sorted by epsilon.

    Demonstrates that tighter privacy (smaller ε) increases metric variance.
    """
    profile = []
    for eps, seed_results in sorted(epsilon_to_seed_results.items()):
        var_summary = dp_training_variance(seed_results)
        if metric not in var_summary:
            continue
        profile.append(
            {
                "epsilon": eps,
                "mean": var_summary[metric]["mean"],
                "std": var_summary[metric]["std"],
                "n_seeds": var_summary[metric]["n_seeds"],
            }
        )
    return profile


# ---------------------------------------------------------------------------
# 3. Significance Testing with Bonferroni Correction
# ---------------------------------------------------------------------------

def _wilcoxon_statistic(differences: list[float]) -> float:
    """Compute the Wilcoxon signed-rank W+ statistic for paired differences."""
    nonzero = [(abs(d), d) for d in differences if d != 0.0]
    if not nonzero:
        return 0.0
    nonzero.sort(key=lambda x: x[0])
    w_plus = 0.0
    for rank, (_, d) in enumerate(nonzero, start=1):
        if d > 0:
            w_plus += rank
    return w_plus


def _normal_approx_p_value(w_stat: float, n: int) -> float:
    """Two-sided p-value for Wilcoxon via normal approximation (n > 10)."""
    if n <= 0:
        return 1.0
    mean_w = n * (n + 1) / 4.0
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if std_w == 0:
        return 1.0
    z = (w_stat - mean_w) / std_w
    # Two-tailed p-value via complementary error function approximation
    p_one_tail = 0.5 * math.erfc(abs(z) / math.sqrt(2))
    return min(1.0, 2.0 * p_one_tail)


def significance_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_comparisons: int = 1,
    alpha: float = 0.05,
) -> dict[str, float | bool | str]:
    """Wilcoxon signed-rank test for paired metric scores.

    Applies Bonferroni correction: adjusted threshold = alpha / n_comparisons.
    Returns a dict with keys: w_stat, p_value, p_adjusted_threshold,
    significant (bool), and test_label.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have equal length")
    if len(scores_a) < 2:
        raise ValueError("At least 2 paired samples are required")

    diffs = [float(a) - float(b) for a, b in zip(scores_a, scores_b)]
    nonzero_diffs = [d for d in diffs if d != 0.0]
    n = len(nonzero_diffs)

    w_stat = _wilcoxon_statistic(nonzero_diffs)
    p_value = _normal_approx_p_value(w_stat, n) if n > 0 else 1.0
    adjusted_alpha = alpha / max(1, n_comparisons)

    return {
        "w_stat": w_stat,
        "p_value": p_value,
        "p_adjusted_threshold": adjusted_alpha,
        "significant": bool(p_value < adjusted_alpha),
        "test_label": "†" if p_value < adjusted_alpha else "",
        "n_pairs": float(len(scores_a)),
        "bonferroni_n": float(n_comparisons),
    }


# ---------------------------------------------------------------------------
# 4. Privacy Accountant Epsilon Verification
# ---------------------------------------------------------------------------

def verify_epsilon_accounting(
    reported_epsilon: float,
    computed_epsilon: float,
    *,
    tolerance: float = 0.01,
) -> dict[str, float | bool | str]:
    """Assert that the reported ε matches the privacy accountant's computed ε.

    Returns a result dict.  ``passed`` is True when
    |reported - computed| <= tolerance.
    """
    delta = abs(reported_epsilon - computed_epsilon)
    passed = delta <= tolerance
    return {
        "reported_epsilon": reported_epsilon,
        "computed_epsilon": computed_epsilon,
        "delta": delta,
        "tolerance": tolerance,
        "passed": passed,
        "message": (
            "OK" if passed
            else (
                f"FAIL: reported ε={reported_epsilon} deviates from computed "
                f"ε={computed_epsilon} by {delta:.4f} (tolerance={tolerance})"
            )
        ),
    }


# ---------------------------------------------------------------------------
# 5. Dataset-Type Stratification
# ---------------------------------------------------------------------------

DatasetType = str  # "tabular" | "time_series" | "text"


def stratified_utility_privacy(
    configs_by_type: dict[DatasetType, list[dict[str, float]]],
) -> dict[DatasetType, dict[str, dict[str, float]]]:
    """Compute per-dataset-type mean ± std for utility, privacy, and fidelity scores.

    *configs_by_type* maps dataset type → list of evaluated config dicts (each
    having at least ``privacy_score``, ``utility_score``, ``fidelity_score``).
    """
    metrics = ["privacy_score", "utility_score", "fidelity_score"]
    report: dict[DatasetType, dict[str, dict[str, float]]] = {}

    for dtype, configs in configs_by_type.items():
        if not configs:
            continue
        type_stats: dict[str, dict[str, float]] = {}
        for metric in metrics:
            vals = [float(c[metric]) for c in configs if metric in c]
            if not vals:
                continue
            n = len(vals)
            mean = sum(vals) / n
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n > 1 else 0.0
            type_stats[metric] = {"mean": mean, "std": std, "n": float(n)}
        report[dtype] = type_stats

    return report
