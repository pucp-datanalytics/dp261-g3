import pandas as pd
import joblib
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate, StratifiedKFold


# ============================================================
# 1. Construcción del preprocesador
# ============================================================

def build_preprocessor(num_cols, cat_cols):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preproc = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    return preproc


# ============================================================
# 2. Entrenamiento de un modelo
# ============================================================

def train_model(model, preprocessor, X_train, y_train):
    pipe = Pipeline([
        ("preprocessing", preprocessor),
        ("clf", model)
    ])
    pipe.fit(X_train, y_train)
    return pipe


# ============================================================
# 3. Evaluación con Cross-Validation
# ============================================================

def evaluate_model(model, preprocessor, X, y, metricas=None, n_splits=5):
    if metricas is None:
        metricas = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    pipe = Pipeline([
        ("preprocessing", preprocessor),
        ("clf", model)
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = cross_validate(pipe, X, y, cv=cv, scoring=metricas)

    resultados = {m: scores[f"test_{m}"].mean() for m in metricas}
    return resultados


# ============================================================
# 4. Guardar y cargar modelos
# ============================================================

def save_model(model, path):
    joblib.dump(model, path)
    print(f"Modelo guardado en: {path}")


def load_model(path):
    return joblib.load(path)


# ============================================================
# 5. Registro de experimentos (CSV)
# ============================================================

def log_experiment(model_name, model, metrics, path="../models/experiments_log.csv"):
    fila = {
        "modelo": model_name,
        "parametros": str(model.get_params()),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([fila])], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([fila])

    df.to_csv(path, index=False)
    print(f"Experimento registrado en {path}")


# ============================================================
# 6. Validación de predicción
# ============================================================

def test_prediction(model_path, X_sample):
    modelo = load_model(model_path)
    pred = modelo.predict(X_sample)
    return pred