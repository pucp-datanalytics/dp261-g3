from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# ROL 2 — Winsorizing por IQR
# ──────────────────────────────────────────────

class IQRWinsorizor(BaseEstimator, TransformerMixin):
    """
    Trata outliers aplicando capping por IQR.
    fit() calcula los límites; transform() aplica el clip.
    """
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.array(X)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X, y=None):
        X = np.array(X, dtype=float).copy()
        for i in range(X.shape[1]):
            X[:, i] = np.clip(X[:, i], self.lower_[i], self.upper_[i])
        return X


# ──────────────────────────────────────────────
# ROL 3 — Feature Engineering
# ──────────────────────────────────────────────

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Crea features derivadas a partir de las columnas numéricas originales.

    Features nuevas:
    - credit_utilization : avg_bill / LIMIT_BAL
    - payment_ratio      : avg_pay / (avg_bill + 1)
    - late_payments      : meses con atraso (PAY_2..PAY_6 > 0)
    - bill_trend         : BILL_AMT6 - BILL_AMT1
    - pay_trend          : PAY_AMT6 - PAY_AMT1
    - risk_score         : LIMIT_BAL * late_payments
    - ability_to_pay     : PAY_AMT1 + PAY_AMT2 + PAY_AMT3
    - log_limit          : log1p(LIMIT_BAL)
    - log_avg_bill       : log1p(avg_bill clipped >= 0)
    - age_group          : binning de AGE en 4 grupos
    """

    NUM_COLS = [
        "LIMIT_BAL", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(X, columns=self.NUM_COLS)

        avg_bill = df[[f"BILL_AMT{i}" for i in range(1, 7)]].mean(axis=1)
        avg_pay  = df[[f"PAY_AMT{i}" for i in range(1, 7)]].mean(axis=1)

        df["credit_utilization"] = avg_bill / df["LIMIT_BAL"]
        df["payment_ratio"]      = avg_pay / (avg_bill + 1)
        df["payment_ratio"]      = df["payment_ratio"].fillna(df["payment_ratio"].median())

        pay_cols = ["PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
        df["late_payments"]  = df[pay_cols].apply(lambda r: (r > 0).sum(), axis=1)
        df["bill_trend"]     = df["BILL_AMT6"] - df["BILL_AMT1"]
        df["pay_trend"]      = df["PAY_AMT6"]  - df["PAY_AMT1"]
        df["risk_score"]     = df["LIMIT_BAL"] * df["late_payments"]
        df["ability_to_pay"] = df["PAY_AMT1"]  + df["PAY_AMT2"] + df["PAY_AMT3"]
        df["log_limit"]      = np.log1p(df["LIMIT_BAL"])
        df["log_avg_bill"]   = np.log1p(avg_bill.clip(lower=0))
        df["age_group"]      = pd.cut(
            df["AGE"], bins=[0, 25, 35, 50, 100], labels=[0, 1, 2, 3]
        ).astype(float)

        return df.values


# ──────────────────────────────────────────────
# Función principal
# ──────────────────────────────────────────────

def build_preprocessor(num_cols, cat_cols):
    """
    Construye el ColumnTransformer con:
    - num_pipe: Imputación + Winsorizing IQR + FeatureEngineering + StandardScaler
    - cat_pipe: Imputación + OneHotEncoder
    """
    num_pipe = Pipeline(steps=[
        ("imp",      SimpleImputer(strategy="median")),
        ("outliers", IQRWinsorizor(factor=1.5)),
        ("features", FeatureEngineer()),
        ("sc",       StandardScaler()),
    ])

    cat_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh",  OneHotEncoder(handle_unknown="ignore")),
    ])

    preproc = ColumnTransformer(transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    return preproc
