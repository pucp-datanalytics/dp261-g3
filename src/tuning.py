import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from models import (
    REPO_ROOT,
    DEFAULT_DATA_PATH,
    load_data,
    build_preprocessor,
    build_final_model,
    build_base_estimators,
    build_single_model_pipeline,
)


MODELS_DIR = REPO_ROOT / "models"
TUNED_DIR = MODELS_DIR / "tuned"
DEFAULT_LOG_PATH = MODELS_DIR / "experiments_log.csv"


def tune_model(
    pipe,
    param_grid,
    X,
    y,
    cv,
    scoring="f1",
    name="",
    sprint=None,
    log_path=None,
    log_to_csv=True,
):
    """
    GridSearchCV sobre ``pipe``; persiste el mejor estimador en ``models/tuned/{name}_tuned.pkl``.
    Opcionalmente registra la fila en ``experiments_log.csv``.
    """
    start = time.time()

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    grid.fit(X, y)

    elapsed = time.time() - start

    TUNED_DIR.mkdir(parents=True, exist_ok=True)
    model_path = TUNED_DIR / f"{name}_tuned.pkl"
    joblib.dump(grid.best_estimator_, model_path)

    rel_path = model_path.relative_to(REPO_ROOT).as_posix()

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sprint": sprint if sprint is not None else "",
        "phase": "grid_search",
        "model": name,
        "best_score_cv": grid.best_score_,
        "best_params": str(grid.best_params_),
        "time_seconds": round(elapsed, 2),
        "path": rel_path,
        "notes": f"GridSearchCV scoring={scoring}",
    }

    if log_to_csv:
        log_experiment(result, log_path=log_path)
    return result


def log_experiment(result, log_path=None):
    """Añade una fila al CSV de experimentos (columnas unificadas por unión de conjuntos)."""
    log_path = Path(log_path or DEFAULT_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame([result])

    if log_path.exists():
        old_log = pd.read_csv(log_path)
        updated_log = pd.concat([old_log, new_row], ignore_index=True)
    else:
        updated_log = new_row

    updated_log.to_csv(log_path, index=False)
    print(f"Experimento registrado en {log_path}")


def _eval_on_holdout(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, y_proba),
    }


def export_sprint4_artifacts(
    data_path=None,
    test_size=0.2,
    random_state=42,
    log_path=None,
):
    """
    Entrena y guarda los tres modelos base tuneados (mismos hparams finales) + Soft Voting final.
    Registra cada paso en ``experiments_log.csv``.

    Requiere que exista el CSV de features (p. ej. ``dvc pull``).

    Returns
    -------
    dict con rutas ``final_model``, ``tuned`` y métricas de test del modelo final.
    """
    csv_path = Path(data_path) if data_path is not None else DEFAULT_DATA_PATH
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el dataset: {csv_path}. Ejecuta `dvc pull` desde la raíz del repo."
        )

    X, y = load_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    preproc = build_preprocessor(X_train)

    TUNED_DIR.mkdir(parents=True, exist_ok=True)
    lp = log_path or DEFAULT_LOG_PATH

    svm, gb, ada = build_base_estimators(random_state=random_state)
    singles = [
        ("svm", svm),
        ("gb", gb),
        ("ada", ada),
    ]

    paths_tuned = {}

    for name, estimator in singles:
        pipe = build_single_model_pipeline(preproc, estimator)
        pipe.fit(X_train, y_train)
        out_path = TUNED_DIR / f"{name}_tuned.pkl"
        joblib.dump(pipe, out_path)
        paths_tuned[name] = out_path.relative_to(REPO_ROOT).as_posix()

        metrics = _eval_on_holdout(pipe, X_test, y_test)
        log_experiment(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "sprint": 4,
                "phase": "tuned_single_fit",
                "model": name,
                "best_score_cv": "",
                "best_params": str(estimator.get_params()),
                "time_seconds": "",
                "path": paths_tuned[name],
                "notes": "Pipeline single-estimator post-hyperparameter tuning (Sprint 4)",
                **metrics,
            },
            log_path=lp,
        )

    final_pipe = build_final_model(preproc, random_state=random_state)
    final_pipe.fit(X_train, y_train)

    final_path = MODELS_DIR / "final_model.pkl"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipe, final_path)

    fm_metrics = _eval_on_holdout(final_pipe, X_test, y_test)

    log_experiment(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "sprint": 4,
            "phase": "final_soft_voting",
            "model": "soft_voting_final",
            "best_score_cv": "",
            "best_params": "FINAL_ESTIMATOR_PARAMS en models.py",
            "time_seconds": "",
            "path": final_path.relative_to(REPO_ROOT).as_posix(),
            "notes": "Soft Voting SVM+GB+Ada; mismo split que PB-14/PB-15",
            **fm_metrics,
        },
        log_path=lp,
    )

    return {
        "final_model": str(final_path.resolve()),
        "tuned": paths_tuned,
        "test_metrics_final": fm_metrics,
    }
