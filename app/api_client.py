"""Inference clients for the Inved PoC.

Two paths (decision 2026-05-30): try the MLflow-served REST API first, fall back to
loading the registered champion in-process. Pure Python — Streamlit caching is wired
in app.py, so this module stays importable/testable without a running Streamlit server.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

MODEL_URI = "models:/inved-house-price@Production"
DEFAULT_API_URL = "http://127.0.0.1:5000"
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _df_to_split_payload(df: pd.DataFrame) -> dict:
    """MLflow `/invocations` dataframe_split payload, NaN -> null (JSON-safe)."""
    safe = df.astype(object).where(pd.notna(df), None)
    return {
        "dataframe_split": {
            "columns": list(safe.columns),
            "data": safe.to_numpy(dtype=object).tolist(),
        }
    }


class MLflowApiClient:
    """Client for a running `mlflow models serve` endpoint."""

    mode = "mlflow-api"

    def __init__(self, url: str = DEFAULT_API_URL):
        self.url = url.rstrip("/")

    @staticmethod
    def is_available(url: str = DEFAULT_API_URL, timeout: float = 0.5) -> bool:
        try:
            return requests.get(f"{url.rstrip('/')}/ping", timeout=timeout).status_code == 200
        except requests.RequestException:
            return False

    def predict(self, df: pd.DataFrame) -> tuple[float, float]:
        """Return (prediction_log_space, http_latency_seconds)."""
        payload = _df_to_split_payload(df)
        t0 = time.perf_counter()
        resp = requests.post(f"{self.url}/invocations", json=payload, timeout=10)
        latency = time.perf_counter() - t0
        resp.raise_for_status()
        out = resp.json()
        preds = out["predictions"] if isinstance(out, dict) else out
        return float(preds[0]), latency


class InProcessClient:
    """Loads the registered champion in-process (fallback when the server is down)."""

    mode = "in-process"

    def __init__(self):
        import mlflow

        mlflow.set_tracking_uri(f"file:{_REPO_ROOT / 'mlruns'}")
        self.model = mlflow.sklearn.load_model(MODEL_URI)

    def predict(self, df: pd.DataFrame) -> tuple[float, float]:
        t0 = time.perf_counter()
        pred = self.model.predict(df)[0]
        return float(pred), time.perf_counter() - t0


def get_client(url: str = DEFAULT_API_URL):
    """Try the served REST API first; fall back to in-process load. Returns the client."""
    if MLflowApiClient.is_available(url):
        return MLflowApiClient(url)
    return InProcessClient()
