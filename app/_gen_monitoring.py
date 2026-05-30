"""One-off: per-neighborhood holdout residuals for the Monitoring equity boxplot.

Replicates 2_data_prep's split to align with results/preds_stacking.npz, then
writes app/monitoring_data.json. Run from repo root:  .venv/bin/python app/_gen_monitoring.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import engineer_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = Path(__file__).resolve().parent

df = pd.read_csv(ROOT / "data" / "train.csv")
df = df.drop(df[(df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)].index)  # outlier rule
df["SalePrice_log"] = np.log1p(df["SalePrice"])
feat = engineer_features(df)
X = feat.drop(columns=["SalePrice", "SalePrice_log", "Id"], errors="ignore")
y = feat["SalePrice_log"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

npz = np.load(ROOT / "results" / "preds_stacking.npz")
preds, y_true = npz["Stacking"].astype(float), npz["y_true"].astype(float)
assert len(X_te) == len(preds), f"len mismatch {len(X_te)} vs {len(preds)}"
assert np.allclose(y_te.values, y_true, atol=1e-6), "split misaligned with preds_stacking!"
print(f"alignment OK — {len(X_te)} holdout rows")

resid_pct = (np.expm1(preds) - np.expm1(y_te.values)) / np.expm1(y_te.values) * 100
res = pd.DataFrame({"Neighborhood": X_te["Neighborhood"].values, "resid": resid_pct})
stats = {}
for n, g in res.groupby("Neighborhood"):
    if len(g) >= 3:
        box = g["resid"].quantile([0, .25, .5, .75, 1.0]).round(2).tolist()
        stats[n] = {"box": box, "n": int(len(g)), "median": round(float(g["resid"].median()), 2)}
out = dict(sorted(stats.items(), key=lambda kv: kv[1]["median"]))
(APP / "monitoring_data.json").write_text(json.dumps({"equity": out}, indent=2, ensure_ascii=False))
meds = [v["median"] for v in out.values()]
print(f"wrote monitoring_data.json — {len(out)} quartiers, median residual range "
      f"[{min(meds):.1f}%, {max(meds):.1f}%]")
