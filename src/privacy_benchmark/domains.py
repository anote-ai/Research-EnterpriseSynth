"""Domain definitions for the 5-domain enterprise DP Pareto study (issue #35).

Each domain captures how DP noise affects utility differently depending on:
  - Schema structure (tabular vs. time-series vs. relational)
  - Temporal dependencies (time-series domains suffer more DP degradation)
  - Relational constraints (FK preservation cost under DP)
  - Regulatory context (drives the relevant compliance tier)

The DP sensitivity multiplier encodes how much faster utility degrades relative
to a plain tabular domain as ε tightens.  Values >1 mean "harder to DP-ify".
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DomainSpec:
    """Complete specification of one enterprise data domain."""

    name: str
    display_name: str
    schema_type: str               # "tabular" | "timeseries" | "relational"
    regulatory_context: str        # e.g. "HIPAA", "GDPR", "SOX", "internal"
    recommended_epsilon_range: tuple[float, float]  # (min_eps, max_eps)
    task: str                      # downstream ML task description
    task_metric: str               # primary evaluation metric
    n_features: int                # approximate schema width
    has_temporal_dependency: bool
    has_foreign_keys: bool

    # How much faster utility degrades vs. plain tabular (1.0 = same)
    dp_sensitivity_multiplier: float

    # No-DP baseline TSTR score (upper bound on utility)
    baseline_tstr: float

    # Representative column types for synthesis notes
    column_types: list[str] = field(default_factory=list)

    def utility_at_epsilon(self, epsilon: float) -> float:
        """Simulate TSTR utility at a given ε for this domain.

        Uses a logistic curve calibrated to the domain's DP sensitivity.
        Higher sensitivity → curve shifts right (needs larger ε for same utility).
        scale = 1.5 * multiplier so that larger multiplier → slower rise → lower utility
        at the same ε.
        """
        scale = 1.5 * self.dp_sensitivity_multiplier
        raw = 0.55 + 0.43 * (1 - 1 / (1 + epsilon / scale))
        # Clamp to baseline
        return min(self.baseline_tstr, max(0.10, raw))

    def privacy_leakage_at_epsilon(self, epsilon: float) -> float:
        """Simulate membership-inference AUC at a given ε (higher = more leakage)."""
        # AUC rises from ~0.52 (near-random) toward 0.82 as ε grows
        return min(0.95, 0.52 + 0.30 * (1 - 1 / (1 + epsilon / 2)))

    def fidelity_at_epsilon(self, epsilon: float) -> float:
        """Simulate statistical fidelity (Wasserstein-based) at a given ε."""
        scale = 2.0 * self.dp_sensitivity_multiplier
        raw = 0.58 + 0.40 * (1 - 1 / (1 + epsilon / scale))
        return min(0.99, max(0.10, raw))

    def downstream_accuracy_at_epsilon(self, epsilon: float) -> float:
        """Downstream model accuracy relative to real-data oracle (0–1)."""
        tstr = self.utility_at_epsilon(epsilon)
        # Downstream accuracy = utility weighted by task complexity
        return tstr * (1.0 - 0.02 * self.dp_sensitivity_multiplier)


# ── 5 Enterprise Domain Specifications ───────────────────────────────────────

DOMAINS: dict[str, DomainSpec] = {
    "tabular_hr": DomainSpec(
        name="tabular_hr",
        display_name="HR / CRM Records",
        schema_type="tabular",
        regulatory_context="GDPR",
        recommended_epsilon_range=(1.0, 2.0),
        task="Employee attrition prediction",
        task_metric="TSTR F1",
        n_features=28,
        has_temporal_dependency=False,
        has_foreign_keys=False,
        dp_sensitivity_multiplier=1.0,
        baseline_tstr=0.98,
        column_types=["categorical", "numeric", "ordinal"],
    ),
    "healthcare_ehr": DomainSpec(
        name="healthcare_ehr",
        display_name="Healthcare EHR",
        schema_type="tabular",
        regulatory_context="HIPAA",
        recommended_epsilon_range=(1.0, 5.0),
        task="30-day readmission prediction",
        task_metric="TSTR AUC-ROC",
        n_features=45,
        has_temporal_dependency=False,
        has_foreign_keys=False,
        dp_sensitivity_multiplier=1.15,  # sensitive demographics raise sensitivity
        baseline_tstr=0.97,
        column_types=["ICD codes", "demographics", "vitals", "medications"],
    ),
    "financial_transactions": DomainSpec(
        name="financial_transactions",
        display_name="Financial Transactions",
        schema_type="timeseries",
        regulatory_context="SOX/PCI-DSS",
        recommended_epsilon_range=(5.0, 10.0),
        task="Fraud detection",
        task_metric="TSTR F1 (fraud class)",
        n_features=32,
        has_temporal_dependency=True,
        has_foreign_keys=False,
        dp_sensitivity_multiplier=1.35,  # temporal correlations amplify DP cost
        baseline_tstr=0.96,
        column_types=["amounts", "timestamps", "merchant_category", "geolocation"],
    ),
    "iot_sensor": DomainSpec(
        name="iot_sensor",
        display_name="IoT Sensor Data",
        schema_type="timeseries",
        regulatory_context="internal",
        recommended_epsilon_range=(2.0, 10.0),
        task="Equipment failure prediction",
        task_metric="TSTR Recall@95%Precision",
        n_features=128,
        has_temporal_dependency=True,
        has_foreign_keys=False,
        dp_sensitivity_multiplier=1.55,  # high-dim time-series worst-case
        baseline_tstr=0.94,
        column_types=["temperature", "vibration", "pressure", "cycle_count"],
    ),
    "ecommerce_relational": DomainSpec(
        name="ecommerce_relational",
        display_name="E-commerce Relational",
        schema_type="relational",
        regulatory_context="GDPR/CCPA",
        recommended_epsilon_range=(2.0, 5.0),
        task="Purchase propensity / LTV prediction",
        task_metric="TSTR AUC-ROC",
        n_features=64,          # aggregated across users, items, transactions
        has_temporal_dependency=True,
        has_foreign_keys=True,
        dp_sensitivity_multiplier=1.70,  # FK preservation + temporal = highest cost
        baseline_tstr=0.95,
        column_types=["user_demographics", "item_categories", "order_amounts", "session_data"],
    ),
}

DOMAIN_NAMES: list[str] = list(DOMAINS.keys())


def get_domain(name: str) -> DomainSpec:
    if name not in DOMAINS:
        raise KeyError(f"Unknown domain '{name}'. Available: {DOMAIN_NAMES}")
    return DOMAINS[name]


def domains_by_schema_type(schema_type: str) -> list[DomainSpec]:
    return [d for d in DOMAINS.values() if d.schema_type == schema_type]


def recommend_epsilon(domain: DomainSpec, max_utility_loss: float) -> float:
    """Return the minimum ε satisfying a utility retention constraint.

    Args:
        domain: Target domain spec.
        max_utility_loss: Maximum acceptable fractional utility loss (0.0–1.0).
            E.g. 0.10 means "must retain at least 90% of baseline utility".

    Returns:
        Minimum ε from EPSILON_VALUES that satisfies the constraint, or
        the largest available ε if constraint cannot be met.
    """
    from privacy_benchmark.config import EPSILON_VALUES

    target = domain.baseline_tstr * (1.0 - max_utility_loss)
    for eps in sorted(EPSILON_VALUES):
        if domain.utility_at_epsilon(eps) >= target:
            return eps
    return max(EPSILON_VALUES)
