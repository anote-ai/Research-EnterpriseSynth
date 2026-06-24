"""Tests for real SDG baseline module (issue #44)."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from baseline.datasets import load_dataset, LOADERS, DatasetSpec
from baseline.evaluator import evaluate, _wasserstein_mean, _correlation_delta
from baseline.synthesizers import SYNTHESIZER_NAMES


# ── Dataset loaders ────────────────────────────────────────────────────────────

class TestDatasetLoaders:
    def test_loader_names_match(self):
        assert set(LOADERS.keys()) == {"adult", "credit_g", "diabetes"}

    def test_load_dataset_invalid_raises(self):
        with pytest.raises(KeyError):
            load_dataset("nonexistent")

    def test_diabetes_loads(self):
        df, spec = load_dataset("diabetes")
        assert isinstance(df, pd.DataFrame)
        assert isinstance(spec, DatasetSpec)
        assert len(df) > 0

    def test_diabetes_has_target(self):
        df, spec = load_dataset("diabetes")
        assert spec.target_col in df.columns

    def test_diabetes_target_binary(self):
        df, spec = load_dataset("diabetes")
        assert set(df[spec.target_col].unique()).issubset({0, 1})

    def test_diabetes_no_missing(self):
        df, spec = load_dataset("diabetes")
        assert df.isnull().sum().sum() == 0

    def test_diabetes_feature_count(self):
        df, spec = load_dataset("diabetes")
        assert spec.n_features == df.shape[1] - 1

    def test_diabetes_row_count(self):
        df, spec = load_dataset("diabetes")
        assert spec.n_rows == len(df)
        assert spec.n_rows > 500

    def test_credit_g_loads(self):
        df, spec = load_dataset("credit_g")
        assert len(df) == 1000

    def test_credit_g_target_binary(self):
        df, spec = load_dataset("credit_g")
        assert set(df[spec.target_col].unique()).issubset({0, 1})

    def test_credit_g_enterprise_domain(self):
        _, spec = load_dataset("credit_g")
        assert "Financial" in spec.enterprise_domain

    def test_diabetes_enterprise_domain(self):
        _, spec = load_dataset("diabetes")
        assert "Healthcare" in spec.enterprise_domain

    def test_spec_task_is_binary_classification(self):
        for name in LOADERS:
            _, spec = load_dataset(name)
            assert spec.task == "binary_classification"


# ── Fidelity helpers ───────────────────────────────────────────────────────────

class TestFidelityMetrics:
    def _make_df(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "a": rng.normal(0, 1, n),
            "b": rng.normal(5, 2, n),
            "c": rng.uniform(0, 10, n),
        })

    def test_wasserstein_identical_zero(self):
        df = self._make_df()
        wd = _wasserstein_mean(df, df.copy())
        assert wd < 0.05

    def test_wasserstein_different_positive(self):
        real = self._make_df(seed=0)
        synth = self._make_df(seed=99) * 5  # very different scale
        wd = _wasserstein_mean(real, synth)
        assert wd > 0.1

    def test_wasserstein_symmetric_approx(self):
        a = self._make_df(seed=1)
        b = self._make_df(seed=2)
        wd_ab = _wasserstein_mean(a, b)
        wd_ba = _wasserstein_mean(b, a)
        assert abs(wd_ab - wd_ba) < 0.05

    def test_correlation_delta_identical_zero(self):
        df = self._make_df()
        cd = _correlation_delta(df, df.copy())
        assert cd < 0.01

    def test_correlation_delta_different_positive(self):
        real = self._make_df(seed=0)
        synth = pd.DataFrame({
            "a": np.random.default_rng(99).normal(0, 1, 200),
            "b": np.random.default_rng(100).normal(0, 1, 200),
            "c": np.random.default_rng(101).normal(0, 1, 200),
        })
        cd = _correlation_delta(real, synth)
        assert cd > 0.0

    def test_correlation_delta_single_column_returns_zero(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        cd = _correlation_delta(df, df.copy())
        assert cd == 0.0


# ── Evaluator ──────────────────────────────────────────────────────────────────

class TestEvaluator:
    def _make_binary_df(self, n=300, seed=42):
        rng = np.random.default_rng(seed)
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(1, 1, n)
        y = (x1 + x2 > 0).astype(int)
        return pd.DataFrame({"x1": x1, "x2": x2, "label": y})

    def test_evaluate_returns_result(self):
        real = self._make_binary_df()
        synth = self._make_binary_df(seed=99)
        result = evaluate(real, synth, "label", "test_ds", "MockSynth")
        assert result.oracle_f1 >= 0.0
        assert result.tstr_f1 >= 0.0

    def test_oracle_f1_bounded(self):
        real = self._make_binary_df()
        synth = self._make_binary_df(seed=1)
        result = evaluate(real, synth, "label", "test", "mock")
        assert 0.0 <= result.oracle_f1 <= 1.0
        assert 0.0 <= result.oracle_auc <= 1.0

    def test_tstr_f1_bounded(self):
        real = self._make_binary_df()
        synth = self._make_binary_df(seed=2)
        result = evaluate(real, synth, "label", "test", "mock")
        assert 0.0 <= result.tstr_f1 <= 1.0

    def test_n_real_train_plus_test_equals_total(self):
        real = self._make_binary_df(n=100)
        synth = self._make_binary_df(n=80, seed=5)
        result = evaluate(real, synth, "label", "test", "mock", test_size=0.2)
        assert result.n_real_train + result.n_real_test == 100

    def test_utility_retention_positive(self):
        real = self._make_binary_df()
        synth = self._make_binary_df(seed=3)
        result = evaluate(real, synth, "label", "test", "mock")
        assert result.utility_retention_f1 >= 0.0

    def test_as_dict_has_required_keys(self):
        real = self._make_binary_df()
        synth = self._make_binary_df(seed=4)
        result = evaluate(real, synth, "label", "test", "mock")
        d = result.as_dict()
        required = {
            "dataset", "synthesizer", "oracle_f1", "oracle_auc",
            "tstr_f1", "tstr_auc", "utility_retention_f1", "utility_retention_auc",
            "wasserstein_mean", "correlation_delta",
            "n_synthetic", "n_real_train", "n_real_test",
        }
        assert required.issubset(set(d.keys()))

    def test_good_synthetic_has_high_retention(self):
        real = self._make_binary_df(n=400, seed=0)
        synth = self._make_binary_df(n=400, seed=1)  # same distribution
        result = evaluate(real, synth, "label", "test", "mock")
        assert result.utility_retention_f1 >= 0.5

    def test_random_synthetic_has_lower_retention(self):
        real = self._make_binary_df(n=400, seed=0)
        rng = np.random.default_rng(99)
        # Completely random synthetic — no signal
        random_synth = pd.DataFrame({
            "x1": rng.uniform(-5, 5, 400),
            "x2": rng.uniform(-5, 5, 400),
            "label": rng.integers(0, 2, 400),
        })
        good_synth = self._make_binary_df(n=400, seed=1)
        r_random = evaluate(real, random_synth, "label", "test", "random")
        r_good = evaluate(real, good_synth, "label", "test", "good")
        assert r_good.utility_retention_f1 >= r_random.utility_retention_f1

    def test_wasserstein_in_result_nonnegative(self):
        real = self._make_binary_df()
        synth = self._make_binary_df(seed=10)
        result = evaluate(real, synth, "label", "test", "mock")
        assert result.wasserstein_mean >= 0.0

    def test_n_synthetic_matches_input(self):
        real = self._make_binary_df(n=200)
        synth = self._make_binary_df(n=150, seed=7)
        result = evaluate(real, synth, "label", "test", "mock")
        assert result.n_synthetic == 150


# ── Synthesizer names ──────────────────────────────────────────────────────────

class TestSynthesizerNames:
    def test_all_three_defined(self):
        assert set(SYNTHESIZER_NAMES) == {"GaussianCopula", "CTGAN", "TVAE"}

    def test_names_are_strings(self):
        for name in SYNTHESIZER_NAMES:
            assert isinstance(name, str)
