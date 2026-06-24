"""SDV synthesizer wrappers: CTGAN, TVAE, GaussianCopula.

Each returns a fitted synthesizer and a sampled DataFrame of the same
length as the training set.
"""
from __future__ import annotations

import pandas as pd


SYNTHESIZER_NAMES = ["GaussianCopula", "CTGAN", "TVAE"]


def _make_metadata(df: pd.DataFrame, target_col: str):
    """Build SDV SingleTableMetadata from a DataFrame."""
    from sdv.metadata import SingleTableMetadata
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)
    return meta


def fit_and_sample(
    synthesizer_name: str,
    train_df: pd.DataFrame,
    target_col: str,
    n_samples: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Fit a synthesizer on train_df and return n_samples synthetic rows."""
    from sdv.single_table import (
        CTGANSynthesizer,
        TVAESynthesizer,
        GaussianCopulaSynthesizer,
    )

    meta = _make_metadata(train_df, target_col)

    if synthesizer_name == "CTGAN":
        synth = CTGANSynthesizer(meta, epochs=100, verbose=False)
    elif synthesizer_name == "TVAE":
        synth = TVAESynthesizer(meta, epochs=100)
    elif synthesizer_name == "GaussianCopula":
        synth = GaussianCopulaSynthesizer(meta)
    else:
        raise ValueError(f"Unknown synthesizer: {synthesizer_name}")

    synth.fit(train_df)
    synthetic_df = synth.sample(num_rows=n_samples)
    # Ensure target column is integer
    if target_col in synthetic_df.columns:
        synthetic_df[target_col] = synthetic_df[target_col].round().astype(int).clip(0, 1)
    return synthetic_df
