from pathlib import Path

import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_PATH = REPO_ROOT / "data" / "processed" / "04_default_credit_features.csv"

# Hiperparámetros finales acordados tras tuning (PB-13 / Sprint 4); usados por Soft Voting.
FINAL_ESTIMATOR_PARAMS = {
    "svm": {"C": 0.515, "kernel": "rbf", "gamma": "scale"},
    "gradient_boosting": {
        "n_estimators": 69,
        "learning_rate": 0.087,
        "max_depth": 3,
    },
    "adaboost": {"n_estimators": 81, "learning_rate": 0.09},
}


def load_data(path=None):
    """
    Carga X, y desde el CSV de features.
    Si ``path`` es None, usa ``DEFAULT_DATA_PATH`` (relativo a la raíz del repo).
    """
    csv_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["ID"])

    target = "default payment next month"
    X = df.drop(columns=[target])
    y = df[target]

    return X, y


def build_preprocessor(X):
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ])


def build_base_estimators(random_state=42):
    """Retorna (svm, gradient_boosting, adaboost) con hiperparámetros finales."""
    p = FINAL_ESTIMATOR_PARAMS
    svm = SVC(**p["svm"], probability=True, random_state=random_state)
    gb = GradientBoostingClassifier(**p["gradient_boosting"], random_state=random_state)
    ada = AdaBoostClassifier(**p["adaboost"], random_state=random_state)
    return svm, gb, ada


def build_final_model(preproc, random_state=42):
    """
    Pipeline completo: preprocesamiento → RandomOverSampler → Soft Voting (SVM+GB+Ada).
    """
    svm, gb, ada = build_base_estimators(random_state=random_state)

    voting = VotingClassifier(
        estimators=[
            ("svm", svm),
            ("gb", gb),
            ("ada", ada),
        ],
        voting="soft",
    )

    return ImbPipeline([
        ("preprocessing", preproc),
        ("sampling", RandomOverSampler(random_state=42)),
        ("model", voting),
    ])


def build_single_model_pipeline(preproc, estimator):
    """Un solo clasificador base + mismo preprocesamiento y oversampling (artefactos `*_tuned.pkl`)."""
    return ImbPipeline([
        ("preprocessing", preproc),
        ("sampling", RandomOverSampler(random_state=42)),
        ("model", estimator),
    ])
