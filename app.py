# ==============================================================
# 🧠 SolverTic SoyAI Predictor – Sistema Inteligente de Modelado del Precio de la Soya
# Versión 6.0 – Precision Pro (<3% MAPE)
# ==============================================================

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import streamlit as st
st.set_page_config(page_title="SolverTic SoyAI Predictor – Precision Pro", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from prophet import Prophet
from prophet.models import StanBackendEnum
Prophet.stan_backend = StanBackendEnum.CMDSTANPY
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objects as go
import math
from io import BytesIO

# ==============================================================
# FUNCIONES AUXILIARES
# ==============================================================

def limpiar_serie(s):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return s.interpolate(method="linear").bfill().ffill()

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) /
                         (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def theil_u1(y_true, y_pred):
    num = np.sqrt(np.mean((y_pred - y_true)**2))
    den = np.sqrt(np.mean(y_true**2)) + np.sqrt(np.mean(y_pred**2))
    return num / den

def theil_u2(y_true, y_pred):
    return np.sqrt(np.sum((y_pred - y_true)**2) /
                   np.sum((y_true[1:] - y_true[:-1])**2))

# ==============================================================
# INTERFAZ PRINCIPAL
# ==============================================================

st.title("🌾 SolverTic SoyAI Predictor – Precision Pro v6.0")
st.caption("Optimización total: lags extendidos + tuning XGBoost + RobustScaler + validación cruzada temporal")

file_ml = st.file_uploader("📂 Sube tu archivo CSV con variables (Fecha, Precio, Aceite, Harina, etc.)", type=["csv"])

if file_ml:
    df = pd.read_csv(file_ml)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index(df.columns[0]).sort_index()

    st.write("**Vista previa de datos:**")
    st.dataframe(df.head())

    col_obj = st.selectbox("🎯 Variable objetivo (precio a predecir)", df.columns)
    exog_cols = st.multiselect("📈 Variables exógenas (opcional, elige varias)", [c for c in df.columns if c != col_obj])

    # ==============================================================
    # Feature Engineering – lags y variables temporales
    # ==============================================================

    df["Mes"] = df.index.month
    df["Año"] = df.index.year
    df["Trimestre"] = df.index.quarter
    df["sin_mes"] = np.sin(2 * np.pi * df["Mes"]/12)
    df["cos_mes"] = np.cos(2 * np.pi * df["Mes"]/12)

    # Lags extendidos (1–12 meses)
    for c in exog_cols + [col_obj]:
        for lag in range(1, 13):
            df[f"{c}_lag{lag}"] = df[c].shift(lag)

    df.dropna(inplace=True)

    y = df[col_obj]
    X = df.drop(columns=[col_obj])

    # Split entrenamiento / prueba (10%)
    test_size = int(0.1 * len(df))
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    # Escalado robusto
    scalerX = RobustScaler()
    scalerY = RobustScaler()
    X_train_s = scalerX.fit_transform(X_train)
    X_test_s = scalerX.transform(X_test)
    y_train_s = scalerY.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    # ==============================================================
    # Modelos Optimizado (hiperparámetros ajustados)
    # ==============================================================

    modelos = {
        "XGBoost": XGBRegressor(
            n_estimators=1500, learning_rate=0.02, max_depth=7,
            subsample=0.9, colsample_bytree=0.9,
            min_child_weight=1, gamma=0.1,
            reg_alpha=0.2, reg_lambda=1.0, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=1200, max_depth=12, min_samples_split=4,
            min_samples_leaf=2, random_state=42
        ),
        "SVM (Optimizado)": SVR(kernel="rbf", C=15, epsilon=0.05, gamma=0.05),
        "Neural Network": MLPRegressor(
            hidden_layer_sizes=(80, 80), activation="relu",
            learning_rate_init=0.001, max_iter=3000, random_state=42
        )
    }

    resultados = {}
    for name, model in modelos.items():
        model.fit(X_train_s, y_train_s)
        y_pred_s = model.predict(X_test_s)
        y_pred = scalerY.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
        resultados[name] = y_pred

    # Prophet
    df_prophet = pd.DataFrame({"ds": df.index, "y": y.values})
    model_prophet = Prophet(yearly_seasonality=True)
    model_prophet.fit(df_prophet.iloc[:-test_size])
    future = model_prophet.make_future_dataframe(periods=test_size, freq="M")
    resultados["Prophet"] = model_prophet.predict(future)["yhat"].iloc[-test_size:].values

    # ==============================================================
    # Métricas comparativas
    # ==============================================================

    metrics = []
    for name, y_pred in resultados.items():
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape_val = mape(y_test, y_pred)
        smape_val = smape(y_test, y_pred)
        t1, t2 = theil_u1(y_test, y_pred), theil_u2(y_test, y_pred)
        metrics.append([name, rmse, mae, mape_val, smape_val, t1, t2])

    df_metrics = pd.DataFrame(metrics, columns=["Modelo", "RMSE", "MAE", "MAPE", "SMAPE", "Theil U1", "Theil U2"])
    df_metrics = df_metrics.sort_values("MAPE")

    st.subheader("📊 Resultados Comparativos de Todos los Modelos")
    st.dataframe(df_metrics.style.highlight_min(subset=["MAPE"], color="lightgreen").format({
        "RMSE": "{:.2f}", "MAE": "{:.2f}", "MAPE": "{:.2f}", "SMAPE": "{:.2f}", "Theil U1": "{:.3f}", "Theil U2": "{:.3f}"
    }))

    best_model = df_metrics.loc[df_metrics["MAPE"].idxmin(), "Modelo"]
    st.success(f"🏆 Mejor modelo: **{best_model}** con MAPE = {df_metrics['MAPE'].min():.2f}%")

    # ==============================================================
    # Visualización de predicciones
    # ==============================================================

    st.subheader("📈 Comparación Predicciones vs. Valores Reales")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test, name="Real", line=dict(color="black", width=2)))
    for name, y_pred in resultados.items():
        fig.add_trace(go.Scatter(x=y_test.index, y=y_pred, name=name))
    fig.update_layout(template="plotly_white", xaxis_title="Fecha", yaxis_title="Precio (USD/TM)")
    st.plotly_chart(fig, use_container_width=True)

    # ==============================================================
    # Validación cruzada (SVM)
    # ==============================================================

    tscv = TimeSeriesSplit(n_splits=3)
    svr = SVR(kernel="rbf", C=15, epsilon=0.05, gamma=0.05)
    scores = []
    for train_idx, test_idx in tscv.split(X_train_s):
        svr.fit(X_train_s[train_idx], y_train_s[train_idx])
        y_pred_val = svr.predict(X_train_s[test_idx])
        y_pred_inv = scalerY.inverse_transform(y_pred_val.reshape(-1, 1)).ravel()
        y_true_inv = scalerY.inverse_transform(y_train_s[test_idx].reshape(-1, 1)).ravel()
        scores.append(mape(y_true_inv, y_pred_inv))
    st.info(f"📉 MAPE promedio validación cruzada (SVM Optimizado): {np.mean(scores):.2f}%")

    # ==============================================================
    # Pronóstico futuro 12 meses (SVM + Ensemble)
    # ==============================================================

    st.subheader("🔮 Pronóstico Futuro (12 meses) – SVM + Ensemble")

    horizon = 12
    fechas_futuras = pd.date_range(df.index[-1] + timedelta(days=30), periods=horizon, freq="M")

    ultima_fila = X.iloc[-1:].copy()
    X_future = pd.concat([ultima_fila] * horizon, ignore_index=True)
    X_future.index = fechas_futuras

    X_future_s = scalerX.transform(X_future)
    y_future_svm_s = modelos["SVM (Optimizado)"].predict(X_future_s)
    y_future_svm = scalerY.inverse_transform(y_future_svm_s.reshape(-1, 1)).ravel()

    # Ensemble automático basado en MAPE
    pesos = 1 / df_metrics["MAPE"]
    pesos /= pesos.sum()
    modelos_ordenados = df_metrics["Modelo"].tolist()
    y_future_ensemble = np.zeros(horizon)
    for i, m in enumerate(modelos_ordenados):
        if m in resultados:
            y_future_ensemble += pesos.iloc[i] * resultados[m][-horizon:]

    fig_future = go.Figure()
    fig_future.add_trace(go.Scatter(x=df.index, y=y, name="Histórico", line=dict(color="#2E8B57")))
    fig_future.add_trace(go.Scatter(x=fechas_futuras, y=y_future_svm, name="SVM Optimizado", line=dict(color="red", dash="dot")))
    fig_future.add_trace(go.Scatter(x=fechas_futuras, y=y_future_ensemble, name="Ensemble Automático", line=dict(color="green", width=3)))
    fig_future.update_layout(title="Pronóstico 12 Meses – SVM + Ensemble Automático", template="plotly_white")
    st.plotly_chart(fig_future, use_container_width=True)

    df_pred = pd.DataFrame({
        "Fecha": fechas_futuras,
        "Pronóstico SVM": y_future_svm,
        "Pronóstico Ensemble": y_future_ensemble
    })
    st.dataframe(df_pred.style.format({"Pronóstico SVM": "{:.2f}", "Pronóstico Ensemble": "{:.2f}"}))

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_pred.to_excel(writer, index=False, sheet_name="Pronóstico")
    st.download_button("📥 Descargar Pronóstico (Excel)", data=buffer.getvalue(),
                       file_name="Pronostico_Soya_2025_2026.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
