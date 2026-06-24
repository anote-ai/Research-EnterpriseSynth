"""Public dataset loaders for enterprise-proxy baseline experiments.

Three use cases:
  UC1 — HR / Attrition proxy   : Adult Income (UCI via sklearn)
  UC2 — Financial / Fraud proxy : Credit-G (UCI via sklearn OpenML)
  UC3 — Healthcare / EHR proxy  : Diabetes PIMA (OpenML id=37)
"""
from __future__ import annotations

import pandas as pd
from dataclasses import dataclass


@dataclass
class DatasetSpec:
    name: str
    display_name: str
    enterprise_domain: str
    target_col: str
    task: str          # "binary_classification"
    positive_label: str
    n_rows: int
    n_features: int


def load_adult() -> tuple[pd.DataFrame, DatasetSpec]:
    """Adult Income — HR / attrition proxy.  Binary: income >50K."""
    from sklearn.datasets import fetch_openml
    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = data.frame.copy()
    df = df.dropna().reset_index(drop=True)
    # Encode target as binary int
    df["income"] = (df["class"].astype(str).str.strip() == ">50K").astype(int)
    df = df.drop(columns=["class"])
    # Encode categoricals as strings so SDV handles them
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(str)
    spec = DatasetSpec(
        name="adult",
        display_name="Adult Income (HR proxy)",
        enterprise_domain="HR / CRM Records",
        target_col="income",
        task="binary_classification",
        positive_label="1",
        n_rows=len(df),
        n_features=df.shape[1] - 1,
    )
    return df, spec


def load_credit_g() -> tuple[pd.DataFrame, DatasetSpec]:
    """Credit-G — financial risk / fraud proxy.  Binary: good/bad credit."""
    from sklearn.datasets import fetch_openml
    data = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    df = data.frame.copy()
    df = df.dropna().reset_index(drop=True)
    df["class"] = (df["class"].astype(str).str.strip() == "good").astype(int)
    df = df.rename(columns={"class": "credit_risk"})
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(str)
    spec = DatasetSpec(
        name="credit_g",
        display_name="Credit-G (Financial proxy)",
        enterprise_domain="Financial Transactions",
        target_col="credit_risk",
        task="binary_classification",
        positive_label="1",
        n_rows=len(df),
        n_features=df.shape[1] - 1,
    )
    return df, spec


def load_diabetes() -> tuple[pd.DataFrame, DatasetSpec]:
    """PIMA Diabetes — healthcare / EHR proxy.  Binary: diabetes onset."""
    from sklearn.datasets import fetch_openml
    data = fetch_openml("diabetes", version=1, as_frame=True, parser="auto")
    df = data.frame.copy()
    df = df.dropna().reset_index(drop=True)
    df["Outcome"] = (df["class"].astype(str).str.strip() == "tested_positive").astype(int)
    df = df.drop(columns=["class"])
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(str)
    spec = DatasetSpec(
        name="diabetes",
        display_name="Diabetes PIMA (Healthcare EHR proxy)",
        enterprise_domain="Healthcare EHR",
        target_col="Outcome",
        task="binary_classification",
        positive_label="1",
        n_rows=len(df),
        n_features=df.shape[1] - 1,
    )
    return df, spec


LOADERS = {
    "adult": load_adult,
    "credit_g": load_credit_g,
    "diabetes": load_diabetes,
}


def load_dataset(name: str) -> tuple[pd.DataFrame, DatasetSpec]:
    if name not in LOADERS:
        raise KeyError(f"Unknown dataset: {name}. Choose from {list(LOADERS)}")
    return LOADERS[name]()
