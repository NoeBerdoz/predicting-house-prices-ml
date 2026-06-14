"""Feature engineering for the Inved PoC.

Mirrors `2_data_prep.ipynb` (`engineer_features`) verbatim, kept standalone so the
app has zero notebook dependency. Since the champion pipeline now folds feature
engineering in (RawFeatureEngineer, NB5 §6.1), the served signature is the RAW
schema: `build_row` emits a single RAW row (79 original columns) — no manual
engineering on the client side. `engineer_features` is kept for the monitoring
charts (which still derive features locally), not for inference.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_DEFAULTS_PATH = Path(__file__).resolve().parent / "defaults.json"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise derived features — identical to the notebook (no fitting, no leakage)."""
    df = df.copy()

    # MSSubClass is a categorical code (20, 60, 75…), not a magnitude → force to string.
    df["MSSubClass"] = df["MSSubClass"].astype(str)

    # Ages at sale time.
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]
    df["GarageAge"] = (df["YrSold"] - df["GarageYrBlt"]).clip(lower=0)

    # Aggregated surfaces / bathrooms.
    df["TotalSF"] = (
        df["1stFlrSF"].fillna(0) + df["2ndFlrSF"].fillna(0) + df["TotalBsmtSF"].fillna(0)
    )
    df["TotalBathrooms"] = (
        df["FullBath"].fillna(0)
        + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )

    # Binary flags.
    df["HasGarage"] = (df["GarageArea"].fillna(0) > 0).astype(int)
    df["HasPool"] = (df["PoolArea"].fillna(0) > 0).astype(int)
    df["HasSecondFloor"] = (df["2ndFlrSF"].fillna(0) > 0).astype(int)

    return df


def load_defaults() -> dict:
    return json.loads(_DEFAULTS_PATH.read_text())


def build_row(form_values: dict, defaults: dict | None = None) -> pd.DataFrame:
    """Advisor form values + training defaults -> 1-row RAW DataFrame in signature order.

    The champion pipeline folds feature engineering in (RawFeatureEngineer), so the
    served signature is the RAW schema: this emits the original columns only, with no
    client-side engineering (the model does it). See NB5 §6.1 (task T43).
    """
    d = defaults if defaults is not None else load_defaults()

    raw = dict(d["raw_defaults"])  # full raw feature row (medians / modes)
    raw.update({k: v for k, v in form_values.items() if v is not None})

    row = pd.DataFrame([raw])

    # Guarantee every signature column is present, in the exact training order.
    for col in d["signature_columns"]:
        if col not in row.columns:
            row[col] = d["raw_defaults"].get(col)
    return row[d["signature_columns"]]
