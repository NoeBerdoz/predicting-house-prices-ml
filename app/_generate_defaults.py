"""One-off: generate app/defaults.json from the training data + champion signature.

Run from repo root:  .venv/bin/python app/_generate_defaults.py
Produces app/defaults.json = {raw_defaults, categorical_options, signature_columns}
and verifies build_row(default) round-trips through the registered champion.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_row  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = Path(__file__).resolve().parent

# --- champion signature (the exact 87 columns the model expects, in order) ---
mlflow.set_tracking_uri(f"file:{ROOT / 'mlruns'}")
info = mlflow.models.get_model_info("models:/inved-house-price@Production")
sig_cols = [c["name"] for c in json.loads(info.signature.inputs.to_json())]
print(f"signature columns: {len(sig_cols)}")

# --- raw defaults from training data ---
train = pd.read_csv(ROOT / "data" / "train.csv")
raw_feature_cols = [c for c in train.columns if c not in ("Id", "SalePrice")]

raw_defaults: dict = {}
categorical_options: dict = {}
for col in raw_feature_cols:
    s = train[col]
    if s.dtype == object:
        # most-frequent value *including* NaN (so 'absent' features default to None)
        top = s.value_counts(dropna=False).idxmax()
        raw_defaults[col] = None if (isinstance(top, float) and pd.isna(top)) else top
        categorical_options[col] = sorted(s.dropna().unique().tolist())
    else:
        raw_defaults[col] = float(np.nanmedian(s)) if s.isna().any() else (
            int(s.median()) if (s % 1 == 0).all() else float(s.median())
        )

defaults = {
    "raw_defaults": raw_defaults,
    "categorical_options": categorical_options,
    "signature_columns": sig_cols,
}
(APP / "defaults.json").write_text(json.dumps(defaults, indent=2, ensure_ascii=False))
print(f"wrote app/defaults.json ({len(raw_defaults)} raw cols, "
      f"{len(categorical_options)} categorical option lists)")

# --- verification: default house round-trips through the champion ---
row = build_row({}, defaults)
assert list(row.columns) == sig_cols, "build_row columns != signature"
print(f"build_row -> {row.shape[1]} cols (matches signature: {list(row.columns) == sig_cols})")

champion = mlflow.sklearn.load_model("models:/inved-house-price@Production")
pred_log = champion.predict(row)[0]
print(f"default-house prediction: log={pred_log:.4f}  ->  ${np.expm1(pred_log):,.0f}")

# sanity: a 'better' house predicts higher
better = build_row({"OverallQual": 10, "GrLivArea": 3000, "KitchenQual": "Ex"}, defaults)
pred_better = champion.predict(better)[0]
print(f"upgraded-house prediction:                 ->  ${np.expm1(pred_better):,.0f}")
assert pred_better > pred_log, "upgraded house should predict higher"
print("ROUND-TRIP OK")
