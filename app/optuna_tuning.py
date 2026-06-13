from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_xgb_pipeline(preprocessor, X, y, n_trials: int = 30, cv: int = 5, random_state: int = 42) -> Tuple[Dict[str, Any], Pipeline, float, optuna.study.Study]:
    """Tune XGB hyperparameters with Optuna and return (best_params, fitted_pipeline, tuning_time_s, study).

    The returned pipeline is fitted on (X, y) with the best parameters.
    """
    X = X.copy()
    y = np.asarray(y).ravel()

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        }
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', XGBRegressor(random_state=random_state, n_jobs=-1, verbosity=0, tree_method='hist', enable_categorical=True, **params)),
        ])
        # negative root mean squared error (sklearn scoring returns negative)
        score = cross_val_score(pipe, X, y, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
        return -float(score)

    study = optuna.create_study(direction='minimize')
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    tuning_time = time.time() - t0

    best_params = study.best_params.copy()
    # build final pipeline with best params and fit on full data
    final_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(random_state=random_state, n_jobs=-1, verbosity=0, tree_method='hist', enable_categorical=True, **best_params)),
    ])
    final_pipe.fit(X, y)

    return best_params, final_pipe, tuning_time, study


def tune_mlp_pipeline(preprocessor, X, y, n_trials: int = 30, cv: int = 5, random_state: int = 42) -> Tuple[Dict[str, Any], Pipeline, float]:
    """Tune an MLPRegressor inside a pipeline using Optuna.

    Returns best_params, fitted_pipeline, tuning_time_s.
    """
    X = X.copy()
    y = np.asarray(y).ravel()

    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 3)
        hidden_layer_sizes = tuple(trial.suggest_int(f'n_units_l{i}', 16, 256, log=True) for i in range(n_layers))
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True)
        alpha = trial.suggest_float('alpha', 1e-6, 1e-2, log=True)
        activation = trial.suggest_categorical('activation', ['relu', 'tanh'])

        mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                           activation=activation,
                           solver='adam',
                           learning_rate_init=learning_rate_init,
                           alpha=alpha,
                           max_iter=2000,
                           early_stopping=True,
                           validation_fraction=0.1,
                           random_state=random_state)

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', mlp),
        ])
        score = cross_val_score(pipe, X, y, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
        return -float(score)

    study = optuna.create_study(direction='minimize')
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    tuning_time = time.time() - t0

    best_params = study.best_params.copy()
    # reconstruct MLP with best params
    n_layers = best_params.get('n_layers', 1)
    hidden_layer_sizes = tuple(best_params[f'n_units_l{i}'] for i in range(n_layers))
    mlp_final = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                             activation=best_params.get('activation', 'relu'),
                             solver='adam',
                             learning_rate_init=best_params.get('learning_rate_init', 1e-3),
                             alpha=best_params.get('alpha', 1e-4),
                             max_iter=2000,
                             early_stopping=True,
                             validation_fraction=0.1,
                             random_state=random_state)
    final_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', mlp_final),
    ])
    final_pipe.fit(X, y)

    return best_params, final_pipe, tuning_time


def tune_svr_pipeline(preprocessor, X, y, n_trials: int = 30, cv: int = 5, random_state: int = 42) -> Tuple[Dict[str, Any], Pipeline, float]:
    """Tune an RBF-SVR inside a pipeline using Optuna."""
    X = X.copy()
    y = np.asarray(y).ravel()

    def objective(trial):
        params = {
            'C': trial.suggest_float('C', 1e-1, 1e3, log=True),
            'gamma': trial.suggest_float('gamma', 1e-4, 1e-1, log=True),
            'epsilon': trial.suggest_float('epsilon', 0.01, 0.3, log=True),
        }
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', SVR(kernel='rbf', **params)),
        ])
        score = cross_val_score(pipe, X, y, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
        return -float(score)

    study = optuna.create_study(direction='minimize')
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    tuning_time = time.time() - t0

    best_params = study.best_params.copy()
    final_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', SVR(kernel='rbf', **best_params)),
    ])
    final_pipe.fit(X, y)
    return best_params, final_pipe, tuning_time

