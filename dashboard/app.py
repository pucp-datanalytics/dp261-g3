import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)

# =============================================================
# CONFIGURACIÓN
# =============================================================
st.set_page_config(page_title="Riesgo Crediticio", page_icon="🛡️", layout="wide")

# =============================================================
# FEATURE ENGINEERING — replica exacta del sprint de features
# =============================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Encoding categóricas
    df["SEX_2"] = (df["SEX"] == 2)
    for i in range(1, 7):
        df[f"EDUCATION_{i}"] = (df["EDUCATION"] == i)
    for i in range(1, 4):
        df[f"MARRIAGE_{i}"] = (df["MARRIAGE"] == i)

    # Ratio utilización de crédito
    df["avg_bill"]           = df[[f"BILL_AMT{i}" for i in range(1, 7)]].mean(axis=1)
    df["credit_utilization"] = df["avg_bill"] / df["LIMIT_BAL"]

    # Ratio de pago
    df["avg_pay"]       = df[[f"PAY_AMT{i}" for i in range(1, 7)]].mean(axis=1)
    df["payment_ratio"] = df["avg_pay"] / (df["avg_bill"] + 1)
    df["payment_ratio"] = df["payment_ratio"].fillna(0)

    # Riesgo de impago
    pay_cols            = ["PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    df["late_payments"] = df[pay_cols].apply(lambda x: (x > 0).sum(), axis=1)

    # Tendencias
    df["bill_trend"] = df["BILL_AMT6"] - df["BILL_AMT1"]
    df["pay_trend"]  = df["PAY_AMT6"]  - df["PAY_AMT1"]

    # Grupos
    df["age_group"]    = pd.cut(df["AGE"], bins=[0, 25, 35, 50, 100],
                                labels=["young", "adult", "middle", "senior"])
    df["credit_level"] = pd.cut(df["LIMIT_BAL"], bins=5)

    # Score de riesgo y capacidad de pago
    df["risk_score"]     = df["LIMIT_BAL"] * df["late_payments"]
    df["ability_to_pay"] = df["PAY_AMT1"] + df["PAY_AMT2"] + df["PAY_AMT3"]

    # Transformaciones logarítmicas
    df["log_limit"]    = np.log1p(df["LIMIT_BAL"])
    df["log_avg_bill"] = np.log1p(df["avg_bill"].clip(lower=0))

    feature_cols = [
        "LIMIT_BAL", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
        "SEX_2",
        "EDUCATION_1", "EDUCATION_2", "EDUCATION_3", "EDUCATION_4", "EDUCATION_5", "EDUCATION_6",
        "MARRIAGE_1", "MARRIAGE_2", "MARRIAGE_3",
        "avg_bill", "credit_utilization", "avg_pay", "payment_ratio",
        "late_payments", "bill_trend", "pay_trend",
        "age_group", "credit_level",
        "risk_score", "ability_to_pay", "log_limit", "log_avg_bill"
    ]
    return df[feature_cols]

# =============================================================
# CARGA DEL MODELO Y DATOS
# =============================================================
TARGET     = "default payment next month"
MODEL_PATH = Path(__file__).parent.parent / "models" / "final_model.pkl"
DATA_PATH  = Path(__file__).parent.parent / "data" / "processed" / "04_default_credit_features.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH).drop(columns=["ID"])
    X  = df.drop(columns=[TARGET])
    y  = df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X, y, X_test, y_test

model                        = load_model()
X_all, y_all, X_test, y_test = load_data()

y_proba = model.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= 0.5).astype(int)

# =============================================================
# HEADER
# =============================================================
st.markdown("## 🛡️ Sistema de Gestión de Riesgo Crediticio")
st.markdown("**Modelo final:** Soft Voting — Sprint 4")
st.divider()

# =============================================================
# SECCIÓN 1 — KPIs
# =============================================================
st.markdown("### 📊 KPIs principales")

acc         = accuracy_score(y_test, y_pred)
prec        = precision_score(y_test, y_pred)
rec         = recall_score(y_test, y_pred)
f1          = f1_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc     = auc(fpr, tpr)

pct_default    = y_all.mean() * 100
pct_no_default = 100 - pct_default

c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
c1.metric("👥 Total clientes", f"{len(y_all):,}")
c2.metric("⚠️ % Default",      f"{pct_default:.1f}%")
c3.metric("✅ % No default",   f"{pct_no_default:.1f}%")
c4.metric("📈 ROC-AUC",        f"{roc_auc:.3f}")
c5.metric("🎯 F1-score",       f"{f1:.3f}")
c6.metric("✔️ Accuracy",       f"{acc:.3f}")
c7.metric("🔍 Precision",      f"{prec:.3f}")
c8.metric("👁️ Recall",         f"{rec:.3f}")

st.divider()

# =============================================================
# SECCIÓN 2 — SIMULADOR DE PREDICCIÓN
# =============================================================
st.markdown("### 🧪 Simulador de predicción")

with st.form("simulador"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Datos personales**")
        LIMIT_BAL = st.number_input("Límite de crédito (USD)", min_value=10000, max_value=1000000, value=50000, step=1000)
        AGE       = st.number_input("Edad", min_value=18, max_value=100, value=35)
        SEX       = st.selectbox("Sexo", options=[1, 2],
                                  format_func=lambda x: "Masculino" if x == 1 else "Femenino")
        EDUCATION = st.selectbox("Educación", options=[1, 2, 3, 4, 5, 6],
                                  format_func=lambda x: {1:"Posgrado", 2:"Universidad",
                                                          3:"Secundaria", 4:"Otro",
                                                          5:"Desconocido", 6:"Desconocido 2"}[x])
        MARRIAGE  = st.selectbox("Estado civil", options=[1, 2, 3],
                                  format_func=lambda x: {1:"Casado", 2:"Soltero", 3:"Otro"}[x])

    with col2:
        st.markdown("**Historial de pagos** (-2=sin uso, -1=puntual, 0=mínimo, 1+=retraso)")
        PAY_0 = st.slider("Pago mes 1", -2, 8, 0)
        PAY_2 = st.slider("Pago mes 2", -2, 8, 0)
        PAY_3 = st.slider("Pago mes 3", -2, 8, 0)
        PAY_4 = st.slider("Pago mes 4", -2, 8, 0)
        PAY_5 = st.slider("Pago mes 5", -2, 8, 0)
        PAY_6 = st.slider("Pago mes 6", -2, 8, 0)

    with col3:
        st.markdown("**Montos de factura y pago (USD)**")
        BILL_AMT1 = st.number_input("Factura mes 1", value=5000)
        BILL_AMT2 = st.number_input("Factura mes 2", value=4800)
        BILL_AMT3 = st.number_input("Factura mes 3", value=4600)
        BILL_AMT4 = st.number_input("Factura mes 4", value=4400)
        BILL_AMT5 = st.number_input("Factura mes 5", value=4200)
        BILL_AMT6 = st.number_input("Factura mes 6", value=4000)
        PAY_AMT1  = st.number_input("Pago mes 1", value=2000)
        PAY_AMT2  = st.number_input("Pago mes 2", value=1800)
        PAY_AMT3  = st.number_input("Pago mes 3", value=1600)
        PAY_AMT4  = st.number_input("Pago mes 4", value=1400)
        PAY_AMT5  = st.number_input("Pago mes 5", value=1200)
        PAY_AMT6  = st.number_input("Pago mes 6", value=1000)

    submitted = st.form_submit_button("🔍 Predecir default", use_container_width=True)

def interpret_pay(value):
    if value == -2:
        return "🟣 Sin uso de crédito"
    elif value == -1:
        return "🟢 Pagó puntual"
    elif value == 0:
        return "🟡 Pago mínimo / normal"
    elif value == 1:
        return "🟠 Atraso leve (1 mes)"
    elif value == 2:
        return "🔴 Atraso moderado (2 meses)"
    elif value >= 3:
        return "🚨 Atraso severo (3+ meses)"
    else:
        return "❓ Desconocido"
if submitted:
    raw = pd.DataFrame([{
        "LIMIT_BAL": LIMIT_BAL, "AGE": AGE, "SEX": SEX,
        "EDUCATION": EDUCATION, "MARRIAGE": MARRIAGE,
        "PAY_0": PAY_0, "PAY_2": PAY_2, "PAY_3": PAY_3,
        "PAY_4": PAY_4, "PAY_5": PAY_5, "PAY_6": PAY_6,
        "BILL_AMT1": BILL_AMT1, "BILL_AMT2": BILL_AMT2, "BILL_AMT3": BILL_AMT3,
        "BILL_AMT4": BILL_AMT4, "BILL_AMT5": BILL_AMT5, "BILL_AMT6": BILL_AMT6,
        "PAY_AMT1": PAY_AMT1, "PAY_AMT2": PAY_AMT2, "PAY_AMT3": PAY_AMT3,
        "PAY_AMT4": PAY_AMT4, "PAY_AMT5": PAY_AMT5, "PAY_AMT6": PAY_AMT6,
    }])

    input_features = build_features(raw)
    prob = model.predict_proba(input_features)[0][1]

    st.markdown("### 💳 Interpretación del historial de pagos")

    st.write({
        "PAY_0 (mes actual)": interpret_pay(PAY_0),
        "PAY_2": interpret_pay(PAY_2),
        "PAY_3": interpret_pay(PAY_3),
        "PAY_4": interpret_pay(PAY_4),
        "PAY_5": interpret_pay(PAY_5),
        "PAY_6": interpret_pay(PAY_6),
    })

    if prob < 0.30:
        st.success(f"✅ **No default** — Probabilidad de mora: {prob:.1%} | 🟢 Riesgo bajo")
    elif prob < 0.60:
        st.warning(f"⚠️ **Riesgo medio** — Probabilidad de mora: {prob:.1%} | 🟡 Riesgo medio")
    else:
        st.error(f"🚨 **Default** — Probabilidad de mora: {prob:.1%} | 🔴 Riesgo alto")

    # Guardar en tabla
    if "predicciones" not in st.session_state:
        st.session_state.predicciones = []

    st.session_state.predicciones.append({
        "Límite crédito": f"${LIMIT_BAL:,}",
        "Edad": AGE,
        "Pagos tardíos": int(input_features["late_payments"].values[0]),
        "Probabilidad": f"{prob:.1%}",
        "Resultado": "Default" if prob >= 0.5 else "No default"
    })

    # =============================================================
    # SECCIÓN 5 — SHAP
    # =============================================================
    st.markdown("### 💡 Explicabilidad SHAP — Esta predicción")

    try:
        preprocessor = model.named_steps["preprocessing"]
        classifier   = model.named_steps["model"]

        input_transformed = preprocessor.transform(input_features)

        X_test_transformed = preprocessor.transform(X_test)
        background = shap.sample(X_test_transformed, 50)

        explainer = shap.Explainer(classifier.predict, background)

        shap_vals = explainer(input_transformed)

        fig, ax = plt.subplots(figsize=(8, 4))

        shap.summary_plot(
            shap_vals.values,
            input_transformed,
            feature_names=preprocessor.get_feature_names_out(),
            plot_type="bar",
            show=False
        )

        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.info(f"SHAP no disponible: {e}")

# =============================================================
# SECCIÓN 4 — MÉTRICAS DEL MODELO
# =============================================================
st.markdown("### 📉 Métricas del modelo")

col_cm, col_roc = st.columns(2)

with col_cm:
    st.markdown("**Matriz de confusión**")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No default", "Default"])
    ax.set_yticklabels(["No default", "Default"])
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_roc:
    st.markdown("**Curva ROC**")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.legend(loc="lower right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
# =============================================================
# SECCIÓN 6 — TABLA DE PREDICCIONES
# =============================================================

st.divider()

st.markdown("### 📋 Tabla de predicciones")

if "predicciones" in st.session_state and len(st.session_state.predicciones) > 0:
    df_pred = pd.DataFrame(st.session_state.predicciones)
    st.dataframe(df_pred, use_container_width=True)

    if st.button("🗑️ Limpiar tabla"):
        st.session_state.predicciones = []
        st.rerun()
else:
    st.info("Aún no hay predicciones. Usa el simulador.")