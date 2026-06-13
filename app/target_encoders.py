from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from typing import Optional, List


class CrossFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoding implemented as a scikit-learn transformer.

    - Works column-wise on object/string categorical columns.
    - During fit_transform on training data it uses CV (out-of-fold) estimates to
      compute encoded values for the training rows (prevents leakage when used
      with cross-validation on the full pipeline).
    - After fit(), transform() maps categories using the mapping learned on the
      whole training set; unseen categories are mapped to the global target mean.

    Parameters
    ----------
    cols : list[str] | None
        Columns to encode. If None, encoder expects a 2D array / DataFrame with
        the columns to encode and will encode all of them.
    cv : int
        Number of folds for the internal out-of-fold encoding. If cv <= 1,
        no internal CV is performed (simple mapping from full training set).
    smoothing : float
        Smoothing parameter to regularize category means toward the prior.
        Larger values produce stronger shrinkage toward the global mean.
    min_samples_leaf : int
        Minimal samples to take category average into account.
    random_state : int | None
        Seed for KFold shuffling.
    """

    def __init__(self, cols: Optional[List[str]] = None, cv: int = 5,
                 smoothing: float = 1.0, min_samples_leaf: int = 1,
                 random_state: Optional[int] = None):
        self.cols = cols
        self.cv = int(cv)
        self.smoothing = float(smoothing)
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = random_state

    def fit(self, X, y=None):
        """Learn mapping from category -> (smoothed) target mean using full X,y.

        X may be a DataFrame or 2D array. We store per-column mappings and the
        global prior mean.
        """
        if y is None:
            raise ValueError("CrossFoldTargetEncoder requires y in fit")

        X_df = self._to_frame(X)
        self.feature_names_in_ = X_df.columns.to_list()
        y = np.asarray(y).ravel().astype(float)
        self.prior_ = float(np.nanmean(y))

        self.mapping_ = {}
        for col in X_df.columns:
            ser = X_df[col].astype(object).where(X_df[col].notna(), None)
            df = pd.DataFrame({'cat': ser, 'y': y})
            stats = df.groupby('cat')['y'].agg(['mean', 'count']).rename(columns={'mean': 'mean', 'count': 'count'})

            # smoothing: weighted average of category mean and prior
            counts = stats['count'].astype(float)
            means = stats['mean'].astype(float)
            smooth = (counts * means + self.smoothing * self.prior_) / (counts + self.smoothing)
            mapping = pd.Series(smooth, index=stats.index).to_dict()
            self.mapping_[col] = {'mapping': mapping, 'default': self.prior_}

        return self

    def fit_transform(self, X, y=None, **fit_params):
        """If cv>1, compute out-of-fold encodings for training rows to avoid leakage.

        Returns a numpy array (n_samples, n_cols).
        """
        if y is None:
            raise ValueError("CrossFoldTargetEncoder requires y in fit_transform")

        X_df = self._to_frame(X)
        y = np.asarray(y).ravel().astype(float)

        # OOF encoding
        if self.cv and self.cv > 1:
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            oof = np.full((len(X_df), len(X_df.columns)), np.nan, dtype=float)

            for tr_idx, val_idx in kf.split(X_df):
                X_tr = X_df.iloc[tr_idx]
                y_tr = y[tr_idx]
                # learn mapping on train split
                mapping_split = {}
                prior = float(np.nanmean(y_tr))
                for i, col in enumerate(X_df.columns):
                    ser = X_tr[col].astype(object).where(X_tr[col].notna(), None)
                    df = pd.DataFrame({'cat': ser, 'y': y_tr})
                    stats = df.groupby('cat')['y'].agg(['mean', 'count']).rename(columns={'mean': 'mean', 'count': 'count'})
                    counts = stats['count'].astype(float)
                    means = stats['mean'].astype(float)
                    smooth = (counts * means + self.smoothing * prior) / (counts + self.smoothing)
                    mapping_split[col] = {'mapping': pd.Series(smooth, index=stats.index).to_dict(), 'default': prior}

                # apply mapping to val_idx rows
                X_val = X_df.iloc[val_idx]
                for i, col in enumerate(X_df.columns):
                    col_vals = X_val[col].astype(object).where(X_val[col].notna(), None).values
                    mapdict = mapping_split[col]['mapping']
                    default = mapping_split[col]['default']
                    oof[val_idx, i] = [mapdict.get(v, default) for v in col_vals]

            # After OOF creation, learn full mapping on all data for use in transform()
            self.fit(X_df, y)
            return oof
        else:
            # no cv: just fit on full data and transform accordingly
            self.fit(X_df, y)
            return self.transform(X_df)

    def transform(self, X):
        X_df = self._to_frame(X)
        out = np.empty((len(X_df), len(X_df.columns)), dtype=float)
        for i, col in enumerate(X_df.columns):
            col_vals = X_df[col].astype(object).where(X_df[col].notna(), None).values
            mapdict = self.mapping_[col]['mapping']
            default = self.mapping_[col]['default']
            out[:, i] = [mapdict.get(v, default) for v in col_vals]
        return out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return [f"te_{c}" for c in input_features]

    def _to_frame(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            # assume numpy array
            X = np.asarray(X)
            if X.ndim == 1:
                df = pd.DataFrame(X, columns=(self.cols or [0]))
            else:
                cols = self.cols if self.cols is not None else list(range(X.shape[1]))
                df = pd.DataFrame(X, columns=cols)
        return df

