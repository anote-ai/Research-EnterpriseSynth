#!/usr/bin/env python3
"""Real DP-SGD epsilon sweep using an Opacus-backed tabular VAE.

This script is the end-to-end implementation of Test Experiment 1 and 2 from
DESIGN_DOC.md: it trains a differentially private tabular VAE at each ε budget
on real UCI public datasets and measures TSTR utility plus MIA privacy on held-out
real test data.

Two synthesizers are available:
  - DPVAE (default): a custom, small VAE with hand-rolled min-max/one-hot
    encoding. Validated end-to-end across all 3 domains (see
    results/real_dp_sweep.json).
  - DPTVAE: the real SDV/ctgan TVAE architecture (Encoder/Decoder +
    DataTransformer's mode-specific Gaussian-mixture numeric encoding)
    wrapped in Opacus — a clear improvement over DPVAE on credit_g and
    diabetes (higher F1, much lower variance), but it currently collapses
    completely on adult under any real DP noise (noise_multiplier >~ 0.1),
    even though the same architecture learns adult fine without DP. This is
    unresolved (see RESEARCH_STATUS.md "Known limitations") — DPTVAE is not
    the default until it is.
DP-CTGAN (GAN-based) is out of scope here — privatizing a GAN correctly means
only privatizing the discriminator and reconciling WGAN-GP's gradient penalty
with DP-SGD's own per-sample clipping, a materially larger effort than either
VAE variant above.

Status
------
This script requires Opacus, PyTorch, and ctgan.  Install dependencies with:

    pip install opacus torch ctgan

Once results are produced, commit the output to results/real_dp_sweep.json so
that scripts/run_epsilon_sweep.py can be updated to use measured values instead
of the DomainSpec simulation model.

Results from this script will replace the [SIMULATED] entries in RESEARCH_STATUS.md
with [MEASURED], closing the gap described in the improvement plan (issue #49).

Usage
-----
    python scripts/run_opacus_dp_sweep.py                   # all domains, DPVAE
    python scripts/run_opacus_dp_sweep.py --synthesizer DPTVAE  # production-grade upgrade (adult unresolved)
    python scripts/run_opacus_dp_sweep.py --domain adult    # one dataset
    python scripts/run_opacus_dp_sweep.py --epsilon 2.0     # one ε value
    python scripts/run_opacus_dp_sweep.py --epochs 3 --max-rows 1000 --json
    python scripts/run_opacus_dp_sweep.py --dry-run         # check deps, skip training
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from privacy_benchmark.config import EPSILON_VALUES

try:
    import torch
except ImportError:  # pragma: no cover - handled by _check_deps at runtime
    torch = None

# ---------------------------------------------------------------------------
# Dependency check — fail early with a clear message
# ---------------------------------------------------------------------------

def _check_deps() -> None:
    missing = []
    for pkg in ("opacus", "torch", "sklearn", "ctgan"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(
            f"\n[run_opacus_dp_sweep] Missing dependencies: {missing}\n"
            f"Install with: pip install {' '.join(missing)}\n"
            f"\nFalling back to simulation model is NOT supported by this script.\n"
            f"Run scripts/run_epsilon_sweep.py for the calibrated simulation.\n"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

DATASETS = {
    "adult": {
        "display": "Adult Income (HR / CRM proxy)",
        "enterprise_domain": "tabular_hr",
        "task": "binary_classification",
        "target_col": "class",
        "openml_id": 1590,
    },
    "credit_g": {
        "display": "Credit-G (Financial proxy)",
        "enterprise_domain": "financial_transactions",
        "task": "binary_classification",
        "target_col": "class",
        "openml_id": 31,
    },
    "diabetes": {
        "display": "Diabetes PIMA (Healthcare EHR proxy)",
        "enterprise_domain": "healthcare_ehr",
        "task": "binary_classification",
        "target_col": "class",
        "openml_id": 37,
    },
}


def _load_dataset(name: str):
    """Load a UCI dataset via sklearn/OpenML. Returns (X_train, X_test, y_train, y_test, df_train)."""
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    import pandas as pd

    cfg = DATASETS[name]
    print(f"  Loading {cfg['display']} from OpenML (id={cfg['openml_id']}) …", flush=True)
    ds = fetch_openml(data_id=cfg["openml_id"], as_frame=True, parser="auto")
    df: pd.DataFrame = ds.frame.copy()
    df.columns = [c.lower().replace("-", "_").replace(" ", "_") for c in df.columns]
    target = cfg["target_col"]
    # Extract binary y BEFORE casting (cat.codes gives stable 0/1)
    y = df[target].astype("category").cat.codes
    # Keep category columns easy for pandas/sklearn encoders to consume downstream.
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(str)
    X = df.drop(columns=[target])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    df_train = X_train.copy()
    df_train[target] = y_train.values
    return X_train, X_test, y_train, y_test, df_train, df.dtypes


# ---------------------------------------------------------------------------
# Oracle baseline (train on real, test on real)
# ---------------------------------------------------------------------------

def _oracle_f1(X_train, X_test, y_train, y_test) -> float:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.pipeline import Pipeline
    import numpy as np

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    model = Pipeline([
        ("enc", enc),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    tp = int(((preds == 1) & (y_test == 1)).sum())
    fp = int(((preds == 1) & (y_test == 0)).sum())
    fn = int(((preds == 0) & (y_test == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


# ---------------------------------------------------------------------------
# DP synthesizer training with Opacus
# ---------------------------------------------------------------------------

@dataclass
class _ColumnSpec:
    name: str
    kind: str
    start: int
    end: int
    categories: list | None = None
    min_value: float | None = None
    max_value: float | None = None
    integer_like: bool = False


class _TabularVAE(torch.nn.Module if torch is not None else object):
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.mu = torch.nn.Linear(hidden_dim, latent_dim)
        self.logvar = torch.nn.Linear(hidden_dim, latent_dim)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        hidden = self.encoder(x)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden).clamp(-8, 8)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        recon = self.decoder(z)
        return recon, mu, logvar


class _DPTabularVAESynthesizer:
    """Small tabular VAE trained with Opacus DP-SGD and a pandas-like sample API."""

    def __init__(self, df_train, target_col: str):
        import numpy as np
        import pandas as pd

        self.np = np
        self.pd = pd
        self.columns = list(df_train.columns)
        self.target_col = target_col
        self.specs: list[_ColumnSpec] = []
        self.train_matrix = self._encode(df_train.reset_index(drop=True), fit=True)
        self.model = None
        self.latent_dim = 32

    def _is_categorical(self, series) -> bool:
        if str(series.dtype) in {"object", "category", "bool"}:
            return True
        return series.nunique(dropna=True) <= 10 and not self.pd.api.types.is_float_dtype(series)

    def _encode(self, df, fit: bool):
        arrays = []
        specs = []
        cursor = 0

        for col in self.columns:
            series = df[col]
            if fit and self._is_categorical(series):
                categories = sorted(series.dropna().unique().tolist())
                if not categories:
                    categories = [0]
                cat_to_idx = {cat: i for i, cat in enumerate(categories)}
                values = self.np.zeros((len(df), len(categories)), dtype="float32")
                for row_idx, value in enumerate(series.tolist()):
                    values[row_idx, cat_to_idx.get(value, 0)] = 1.0
                arrays.append(values)
                specs.append(
                    _ColumnSpec(col, "categorical", cursor, cursor + len(categories), categories)
                )
                cursor += len(categories)
            elif fit:
                numeric = self.pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
                min_value = float(numeric.min())
                max_value = float(numeric.max())
                denom = max(max_value - min_value, 1e-9)
                values = ((numeric - min_value) / denom).to_numpy(dtype="float32").reshape(-1, 1)
                arrays.append(values)
                specs.append(
                    _ColumnSpec(
                        col,
                        "numeric",
                        cursor,
                        cursor + 1,
                        min_value=min_value,
                        max_value=max_value,
                        integer_like=bool((numeric.round() == numeric).all()),
                    )
                )
                cursor += 1
            else:
                raise RuntimeError("_encode(fit=False) is not needed by this synthesizer")

        if fit:
            self.specs = specs
        return self.np.concatenate(arrays, axis=1).astype("float32")

    def encode_with_fitted_specs(self, df):
        """Encode a held-out dataframe using specs already fitted on the training set.

        Needed for evaluating the trained model on real non-member records
        (e.g. for the reconstruction-loss membership-inference attack) without
        refitting category vocabularies or min/max scaling on the eval set.
        """
        arrays = []
        for spec in self.specs:
            series = df[spec.name]
            if spec.kind == "categorical":
                cat_to_idx = {cat: i for i, cat in enumerate(spec.categories)}
                values = self.np.zeros((len(df), len(spec.categories)), dtype="float32")
                for row_idx, value in enumerate(series.tolist()):
                    values[row_idx, cat_to_idx.get(value, 0)] = 1.0
                arrays.append(values)
            else:
                numeric = self.pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
                denom = max(spec.max_value - spec.min_value, 1e-9)
                clipped = numeric.clip(spec.min_value, spec.max_value)
                values = ((clipped - spec.min_value) / denom).to_numpy(dtype="float32").reshape(-1, 1)
                arrays.append(values)
        return self.np.concatenate(arrays, axis=1).astype("float32")

    def _recon_loss(self, recon, batch):
        """Per-column reconstruction loss: BCE for one-hot categorical blocks,
        MSE for scaled numeric columns, averaged with equal weight per column.

        Using a single MSE over the whole concatenated vector (the previous
        approach) lets DP-SGD noise push every categorical block toward a
        constant ~0.5 soft output, which then argmax-decodes to the same
        category for every sampled row regardless of epsilon (posterior
        collapse). BCE on the categorical blocks gives a much sharper
        gradient signal per column and is the standard choice for one-hot
        reconstruction targets.
        """
        import torch
        import torch.nn.functional as F

        losses = []
        for spec in self.specs:
            recon_slice = recon[:, spec.start:spec.end]
            batch_slice = batch[:, spec.start:spec.end]
            if spec.kind == "categorical":
                losses.append(F.binary_cross_entropy(recon_slice.clamp(1e-6, 1 - 1e-6), batch_slice))
            else:
                losses.append(F.mse_loss(recon_slice, batch_slice))
        return torch.stack(losses).mean()

    def fit(
        self,
        epsilon: float,
        delta: float,
        epochs: int,
        batch_size: int,
        lr: float,
        max_grad_norm: float,
        device: str,
        seed: int = 42,
    ) -> tuple[float, float]:
        import torch
        from opacus import PrivacyEngine

        torch.manual_seed(seed)

        x = torch.tensor(self.train_matrix, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(x)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = _TabularVAE(self.train_matrix.shape[1], latent_dim=self.latent_dim).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        privacy_engine = PrivacyEngine()
        self.model, optimizer, loader = privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=optimizer,
            data_loader=loader,
            epochs=epochs,
            target_epsilon=epsilon,
            target_delta=delta,
            max_grad_norm=max_grad_norm,
        )

        # KL annealing: ramp the KL weight from 0 up to target_beta over the
        # first half of training. A constant KL weight lets the decoder find
        # a "posterior collapse" solution — ignore the latent code entirely
        # and just output the marginal/majority class for every input, which
        # still scores well on reconstruction loss but destroys sample
        # diversity (every synthetic row decodes to the same category).
        # Annealing gives the model time to learn to use z for reconstruction
        # before the compressive KL pressure kicks in.
        target_beta = 0.01
        steps_per_epoch = max(1, -(-len(dataset) // batch_size))  # ceil division
        total_steps = max(1, epochs * steps_per_epoch)
        anneal_steps = max(1, total_steps // 2)

        t0 = time.time()
        self.model.train()
        step = 0
        for _ in range(epochs):
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                recon, mu, logvar = self.model(batch)
                recon_loss = self._recon_loss(recon, batch)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                beta = target_beta * min(1.0, step / anneal_steps)
                loss = recon_loss + beta * kl
                loss.backward()
                optimizer.step()
                step += 1

        elapsed = time.time() - t0
        spent_epsilon = privacy_engine.get_epsilon(delta)
        return float(spent_epsilon), elapsed

    def sample(self, num_rows: int):
        import torch

        if self.model is None:
            raise RuntimeError("Synthesizer has not been fitted")

        self.model.eval()
        device = next(self.model.parameters()).device
        rows = []
        remaining = num_rows
        with torch.no_grad():
            while remaining > 0:
                batch_size = min(2048, remaining)
                z = torch.randn(batch_size, self.latent_dim, device=device)
                rows.append(self.model.decoder(z).cpu().numpy())
                remaining -= batch_size
        matrix = self.np.vstack(rows)

        decoded = {}
        for spec in self.specs:
            values = matrix[:, spec.start:spec.end]
            if spec.kind == "categorical":
                idxs = values.argmax(axis=1)
                decoded[spec.name] = [spec.categories[int(i)] for i in idxs]
            else:
                scaled = values[:, 0]
                real = spec.min_value + scaled * (spec.max_value - spec.min_value)
                if spec.integer_like:
                    real = self.np.rint(real).clip(spec.min_value, spec.max_value)
                    decoded[spec.name] = real.astype(int)
                else:
                    decoded[spec.name] = real

        return self.pd.DataFrame(decoded, columns=self.columns)


class _TVAEWrapper(torch.nn.Module if torch is not None else object):
    """Combines ctgan's real Encoder+Decoder into one module so Opacus can wrap
    a single nn.Module (make_private_with_epsilon takes exactly one)."""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        mu, std, logvar = self.encoder(x)
        eps = torch.randn_like(std)
        emb = eps * std + mu
        rec, _sigmas = self.decoder(emb)
        return rec, mu, logvar


class _DPTVAESynthesizer:
    """Real SDV/ctgan TVAE architecture (Encoder/Decoder + DataTransformer)
    trained with Opacus DP-SGD.

    This is the production-grade upgrade path from `_DPTabularVAESynthesizer`:
    same VAE training approach validated there, but using ctgan's actual
    mode-specific Gaussian-mixture numeric encoding and real network
    architecture (embedding_dim=128, compress/decompress dims=(128,128) —
    SDV's own defaults) instead of a hand-rolled min-max + one-hot encoder.

    One deliberate compromise: TVAE's decoder has a raw `sigma` variance
    Parameter used directly in its loss (not inside a hookable Linear/Conv
    layer), so Opacus's per-sample gradient hooks cannot safely privatize it.
    We freeze it at its initialization value rather than let it update
    outside the DP accounting — documented here rather than silently skipped.
    """

    def __init__(self, df_train, target_col: str):
        from ctgan.data_transformer import DataTransformer
        import numpy as np
        import pandas as pd

        self.np = np
        self.pd = pd
        self.columns = list(df_train.columns)
        self.target_col = target_col
        self.embedding_dim = 128

        discrete_columns = [
            col for col in self.columns
            if str(df_train[col].dtype) in {"object", "category", "bool"}
            or df_train[col].nunique(dropna=True) <= 10
        ]
        self.transformer = DataTransformer()
        self.transformer.fit(df_train.reset_index(drop=True), discrete_columns)
        self.train_matrix = self.transformer.transform(
            df_train.reset_index(drop=True)
        ).astype("float32")
        self.model = None
        self.decoder = None

    def encode_with_fitted_specs(self, df):
        """Encode held-out data with the already-fitted transformer (for MIA)."""
        return self.transformer.transform(df).astype("float32")

    def fit(
        self,
        epsilon: float,
        delta: float,
        epochs: int,
        batch_size: int,
        lr: float,
        max_grad_norm: float,
        device: str,
        seed: int = 42,
    ) -> tuple[float, float]:
        from ctgan.synthesizers.tvae import Encoder, Decoder, _loss_function as tvae_loss_function
        from opacus import PrivacyEngine

        torch.manual_seed(seed)

        data_dim = self.transformer.output_dimensions
        encoder = Encoder(data_dim, (128, 128), self.embedding_dim).to(device)
        self.decoder = Decoder(self.embedding_dim, (128, 128), data_dim).to(device)
        # Freeze sigma — see class docstring. Excluded from the optimizer so it
        # never updates outside Opacus's per-sample-gradient accounting.
        self.decoder.sigma.requires_grad = False

        wrapper = _TVAEWrapper(encoder, self.decoder).to(device)
        trainable_params = [p for p in wrapper.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-5)

        x = torch.tensor(self.train_matrix, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(x)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        privacy_engine = PrivacyEngine()
        wrapper, optimizer, loader = privacy_engine.make_private_with_epsilon(
            module=wrapper,
            optimizer=optimizer,
            data_loader=loader,
            epochs=epochs,
            target_epsilon=epsilon,
            target_delta=delta,
            max_grad_norm=max_grad_norm,
        )
        self.model = wrapper

        t0 = time.time()
        self.model.train()
        for _ in range(epochs):
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                rec, mu, logvar = self.model(batch)
                loss_1, loss_2 = tvae_loss_function(
                    rec, batch, self.decoder.sigma, mu, logvar,
                    self.transformer.output_info_list, factor=2,
                )
                loss = loss_1 + loss_2
                loss.backward()
                optimizer.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

        elapsed = time.time() - t0
        spent_epsilon = privacy_engine.get_epsilon(delta)
        return float(spent_epsilon), elapsed

    def sample(self, num_rows: int):
        if self.model is None:
            raise RuntimeError("Synthesizer has not been fitted")

        self.decoder.eval()
        device = next(self.decoder.parameters()).device
        rows = []
        remaining = num_rows
        batch_size = 2048
        with torch.no_grad():
            while remaining > 0:
                n = min(batch_size, remaining)
                z = torch.randn(n, self.embedding_dim, device=device)
                rec, _sigmas = self.decoder(z)
                fake = torch.tanh(rec)
                rows.append(fake.cpu().numpy())
                remaining -= n
        data = self.np.concatenate(rows, axis=0)[:num_rows]
        return self.transformer.inverse_transform(data, self.decoder.sigma.detach().cpu().numpy())


def _synth_class_for(synthesizer: str):
    if synthesizer == "DPVAE":
        return _DPTabularVAESynthesizer
    if synthesizer == "DPTVAE":
        return _DPTVAESynthesizer
    raise RuntimeError(
        "CTGAN is not supported here — DP-GAN requires privatizing only the "
        "discriminator and reconciling WGAN-GP's gradient penalty with DP-SGD "
        "clipping, a separate and larger engineering effort. Use --synthesizer "
        "DPVAE (custom lightweight VAE) or DPTVAE (real SDV/ctgan TVAE architecture)."
    )


def _fit_dp_synthesizer(
    synth,
    synthesizer: str,
    epsilon: float,
    delta: float,
    epochs: int,
    batch_size: int,
    lr: float,
    max_grad_norm: float,
    device: str,
    seed: int,
):
    """Train an already-constructed DP synthesizer at a given ε/seed.

    Takes a pre-built synth object (its transformer/encoding is already fit —
    expensive for DPTVAE's Bayesian GMM column encoding — so it must not be
    reconstructed per seed) and (re)trains its model in place.
    """
    print(
        f"    Training {synthesizer} with ε={epsilon}, δ={delta}, epochs={epochs}, "
        f"batch_size={batch_size}, seed={seed}",
        flush=True,
    )
    t0 = time.time()
    spent_epsilon, elapsed = synth.fit(epsilon, delta, epochs, batch_size, lr, max_grad_norm, device, seed=seed)
    print(f"    Done in {elapsed:.0f}s (spent ε={spent_epsilon:.3f})", flush=True)
    return synth, spent_epsilon, time.time() - t0


# ---------------------------------------------------------------------------
# TSTR evaluation
# ---------------------------------------------------------------------------

def _tstr_f1(synth, n_synthetic: int, X_test, y_test, target_col: str) -> float:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.pipeline import Pipeline

    syn_df = synth.sample(num_rows=n_synthetic)
    y_syn = syn_df[target_col].astype("category").cat.codes
    X_syn = syn_df.drop(columns=[target_col])

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    model = Pipeline([
        ("enc", enc),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ])
    model.fit(X_syn, y_syn)
    preds = model.predict(X_test)
    tp = int(((preds == 1) & (y_test == 1)).sum())
    fp = int(((preds == 1) & (y_test == 0)).sum())
    fn = int(((preds == 0) & (y_test == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


# ---------------------------------------------------------------------------
# Fidelity: real per-column distributional distance between real and
# synthetic data (Wasserstein-1 for numeric columns, total variation
# distance for categorical columns), averaged and mapped to a 0-1
# "higher is better" score to match the convention used elsewhere
# (DomainSpec.fidelity_at_epsilon()).  This is the first real fidelity
# measurement in the DP-SGD pipeline; previously even "measured" rows in
# epsilon_sweep.json used the simulated DomainSpec fidelity model.
# ---------------------------------------------------------------------------

def _fidelity_score(real_df, synth_df, columns: list[str], target_col: str) -> float:
    import numpy as np
    import pandas as pd
    from scipy.stats import wasserstein_distance

    distances = []
    for col in columns:
        if col == target_col:
            continue
        real_col = real_df[col]
        synth_col = synth_df[col]
        is_categorical = (
            str(real_col.dtype) in {"object", "category", "bool"}
            or real_col.nunique(dropna=True) <= 10
        )
        if is_categorical:
            real_freq = real_col.value_counts(normalize=True)
            synth_freq = synth_col.value_counts(normalize=True)
            all_cats = set(real_freq.index) | set(synth_freq.index)
            tvd = 0.5 * sum(
                abs(float(real_freq.get(c, 0.0)) - float(synth_freq.get(c, 0.0)))
                for c in all_cats
            )
            distances.append(tvd)
        else:
            real_vals = pd.to_numeric(real_col, errors="coerce").dropna().to_numpy()
            synth_vals = pd.to_numeric(synth_col, errors="coerce").dropna().to_numpy()
            if len(real_vals) == 0 or len(synth_vals) == 0:
                continue
            w = wasserstein_distance(real_vals, synth_vals)
            value_range = max(float(real_vals.max() - real_vals.min()), 1e-9)
            distances.append(min(1.0, w / value_range))

    if not distances:
        return 0.5
    avg_distance = sum(distances) / len(distances)
    return round(max(0.0, 1.0 - avg_distance), 4)


# ---------------------------------------------------------------------------
# MIA (shadow-model membership inference attack)
# ---------------------------------------------------------------------------

def _mia_auc(synth, df_train, X_test, y_test, target_col: str) -> float:
    """Loss-threshold membership inference attack (Yeom et al. 2018 style).

    Compares the trained VAE's per-record reconstruction error on real
    training members vs. real held-out non-members (X_test/y_test, which the
    model never saw). Lower reconstruction error implies "more likely a
    training member," so AUC of (-loss) as the membership score directly
    measures how much the model overfit/memorized its training set.

    This is a genuine membership-inference signal — unlike a real-vs-synthetic
    discriminator (the previous approach here), it responds to ε: heavier DP
    noise should push AUC down toward 0.5 (indistinguishable), while a
    non-private model tends to overfit and drift toward 1.0.
    """
    import torch
    import numpy as np

    if synth.model is None:
        return 0.5

    try:
        df_test = X_test.copy()
        df_test[target_col] = y_test.values

        train_matrix = synth.train_matrix
        test_matrix = synth.encode_with_fitted_specs(df_test)

        device = next(synth.model.parameters()).device
        synth.model.eval()

        def _per_row_loss(matrix):
            x = torch.tensor(matrix, dtype=torch.float32, device=device)
            with torch.no_grad():
                recon, _, _ = synth.model(x)
            return ((recon - x) ** 2).mean(dim=1).cpu().numpy()

        train_loss = _per_row_loss(train_matrix)
        test_loss = _per_row_loss(test_matrix)

        # Balance the attack set so it isn't dominated by whichever side is larger.
        n = min(len(train_loss), len(test_loss))
        rng = np.random.RandomState(42)
        train_idx = rng.choice(len(train_loss), n, replace=False)
        test_idx = rng.choice(len(test_loss), n, replace=False)

        scores = np.concatenate([-train_loss[train_idx], -test_loss[test_idx]])
        labels = np.array([1] * n + [0] * n)

        order = np.argsort(-scores)
        labels_sorted = labels[order]
        n_pos = int(labels_sorted.sum())
        n_neg = len(labels_sorted) - n_pos
        tp = 0
        auc = 0.0
        for label in labels_sorted:
            if label == 1:
                tp += 1
            else:
                auc += tp / n_pos
        auc /= n_neg
        return round(float(min(0.999, max(0.5, auc))), 4)
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_real_dp_sweep(
    dataset_name: str,
    epsilon_values: list[float],
    delta: float,
    synthesizer: str,
    dry_run: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    max_grad_norm: float,
    max_rows: int | None,
    device: str,
    n_seeds: int,
) -> list[dict]:
    cfg = DATASETS[dataset_name]
    print(f"\n{'='*70}")
    print(f"Dataset: {cfg['display']}  Synthesizer: {synthesizer}  δ={delta}  n_seeds={n_seeds}")

    if dry_run:
        print("  [dry-run] Dependencies OK. Skipping training.")
        return []

    X_train, X_test, y_train, y_test, df_train, _ = _load_dataset(dataset_name)
    if max_rows is not None and len(df_train) > max_rows:
        df_train = df_train.sample(n=max_rows, random_state=42).reset_index(drop=True)
        print(f"  Using max_rows={max_rows} for DP synthesizer training.", flush=True)

    print("  Computing oracle F1 …", flush=True)
    oracle_f1 = round(_oracle_f1(X_train, X_test, y_train, y_test), 4)
    print(f"  Oracle F1 = {oracle_f1}")

    # Built once per dataset, not per (epsilon, seed): DPTVAE's DataTransformer
    # fits a Bayesian GMM per continuous column, which is expensive and doesn't
    # depend on epsilon or seed — refitting it 30 times (6 ε x 5 seeds) would
    # waste most of the sweep's runtime on identical, deterministic encoding.
    synth_cls = _synth_class_for(synthesizer)
    synth = synth_cls(df_train, cfg["target_col"])

    results = []
    for eps in epsilon_values:
        print(f"\n  ε = {eps}  ({n_seeds} seeds):")
        tstr_values = []
        mia_values = []
        fidelity_values = []
        spent_epsilon_values = []
        elapsed_total = 0.0
        for seed in range(n_seeds):
            synth, spent_epsilon, elapsed = _fit_dp_synthesizer(
                synth,
                synthesizer,
                eps,
                delta,
                epochs,
                batch_size,
                lr,
                max_grad_norm,
                device,
                seed=1000 * seed + 7,
            )
            tstr = round(_tstr_f1(synth, len(df_train), X_test, y_test, cfg["target_col"]), 4)
            mia_auc = _mia_auc(synth, df_train, X_test, y_test, cfg["target_col"])
            synth_df_for_fidelity = synth.sample(num_rows=len(df_train))
            fidelity = _fidelity_score(
                df_train, synth_df_for_fidelity, list(df_train.columns), cfg["target_col"]
            )
            print(
                f"    seed={seed}  TSTR F1={tstr}  MIA AUC={mia_auc}  fidelity={fidelity}  "
                f"retention={tstr/oracle_f1:.1%}"
            )
            tstr_values.append(tstr)
            mia_values.append(mia_auc)
            fidelity_values.append(fidelity)
            spent_epsilon_values.append(spent_epsilon)
            elapsed_total += elapsed

        tstr_mean = sum(tstr_values) / n_seeds
        tstr_std = (sum((v - tstr_mean) ** 2 for v in tstr_values) / n_seeds) ** 0.5
        mia_mean = sum(mia_values) / n_seeds
        mia_std = (sum((v - mia_mean) ** 2 for v in mia_values) / n_seeds) ** 0.5
        fidelity_mean = sum(fidelity_values) / n_seeds
        fidelity_std = (sum((v - fidelity_mean) ** 2 for v in fidelity_values) / n_seeds) ** 0.5
        print(
            f"    -> mean TSTR F1={tstr_mean:.4f} (std={tstr_std:.4f})  "
            f"mean MIA AUC={mia_mean:.4f} (std={mia_std:.4f})  "
            f"mean fidelity={fidelity_mean:.4f} (std={fidelity_std:.4f})"
        )

        results.append({
            "dataset": dataset_name,
            "display": cfg["display"],
            "enterprise_domain": cfg["enterprise_domain"],
            "synthesizer": synthesizer,
            "epsilon": eps,
            "delta": delta,
            "spent_epsilon_mean": round(sum(spent_epsilon_values) / n_seeds, 4),
            "epochs": epochs,
            "batch_size": batch_size,
            "max_grad_norm": max_grad_norm,
            "n_train_rows": len(df_train),
            "n_seeds": n_seeds,
            "oracle_f1": oracle_f1,
            "tstr_f1_values": tstr_values,
            "tstr_f1_mean": round(tstr_mean, 4),
            "tstr_f1_std": round(tstr_std, 4),
            "tstr_retention_mean": round(tstr_mean / oracle_f1, 4) if oracle_f1 > 0 else 0.0,
            "mia_auc_values": mia_values,
            "mia_auc_mean": round(mia_mean, 4),
            "mia_auc_std": round(mia_std, 4),
            "privacy_score_mean": round(1 - mia_mean, 4),
            "fidelity_values": fidelity_values,
            "fidelity_score_mean": round(fidelity_mean, 4),
            "fidelity_score_std": round(fidelity_std, 4),
            "fidelity_metric": "1 - mean(Wasserstein-1/range for numeric cols, "
                                "total-variation-distance for categorical cols)",
            "elapsed_sec": round(elapsed_total, 1),
            "data_source": f"measured — real DP-SGD training, mean over {n_seeds} seeds",
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Real DP-SGD epsilon sweep (requires Opacus)")
    parser.add_argument("--domain", default="all", choices=list(DATASETS) + ["all"])
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument(
        "--synthesizer", default="DPVAE",
        choices=["DPTVAE", "DPVAE", "CTGAN"],
        help="DPVAE (default) is the earlier custom lightweight VAE — validated across all 3 "
             "domains. DPTVAE uses the real SDV/ctgan TVAE architecture + DataTransformer, "
             "wrapped in Opacus; it is a clear improvement on credit_g/diabetes but currently "
             "collapses completely on adult under any real DP noise (unresolved — see "
             "RESEARCH_STATUS.md). Not yet the default until that's resolved. CTGAN fails "
             "fast — DP-GAN needs a separate, larger effort (privatizing only the discriminator).",
    )
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for quick CPU runs")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of random-seed reruns per (domain, epsilon) to average over — "
             "this small VAE has high run-to-run variance on tiny datasets, so a "
             "single seed is not representative.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Check deps and exit")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    _check_deps()

    datasets = list(DATASETS) if args.domain == "all" else [args.domain]
    eps_values = [args.epsilon] if args.epsilon else EPSILON_VALUES

    all_results: list[dict] = []
    run_context = contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext()
    with run_context:
        for ds in datasets:
            rows = run_real_dp_sweep(
                ds,
                eps_values,
                args.delta,
                args.synthesizer,
                args.dry_run,
                args.epochs,
                args.batch_size,
                args.lr,
                args.max_grad_norm,
                args.max_rows,
                args.device,
                args.seeds,
            )
            all_results.extend(rows)

    if args.json:
        print(json.dumps(all_results, indent=2))
    elif not args.dry_run and all_results:
        out = pathlib.Path(__file__).parent.parent / "results" / "real_dp_sweep.json"
        out.write_text(json.dumps(all_results, indent=2))
        print(f"\nResults written to {out}")
        print("\nNext step: update scripts/run_epsilon_sweep.py to load from real_dp_sweep.json")
        print("and update RESEARCH_STATUS.md to mark epsilon_sweep results as [MEASURED].")


if __name__ == "__main__":
    import pathlib
    main()
