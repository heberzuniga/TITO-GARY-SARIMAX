# ==============================================================
# 🧠 SolverTic SoyAI Predictor – Sistema Inteligente de Modelado del Precio de la Soya
# Versión 6.5.2 – Corrección Fechas + Checkbox Barras de Error + División Visual
# ==============================================================

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import streamlit as st
st.set_page_config(page_title="SolverTic SoyAI Predictor – Periodos Divididos", layout="wide")

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
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

# ==============================================================
# INTERFAZ PRINCIPAL
# ==============================================================

st.title("🌾 SolverTic SoyAI Predictor – Visualización por Periodos v6.5.2")
st.caption("Entrenamiento, evaluación y pronóstico claramente divididos con opción de mostrar barras de error y tabla comparativa.")

file_ml = st.file_uploader("📂 Sube tu archivo CSV (Fecha, Precio, Aceite, Harina, etc.)", type=["csv"])

if file_ml:
    df = pd.read_csv(file_ml)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index(df.columns[0]).sort_index()

    col_obj = st.selectbox("🎯 Variable objetivo (precio a predecir)", df.columns)
    exog_cols = st.multiselect("📈 Variables exógenas (opcional)", [c for c in df.columns if c != col_obj])
    show_errors = st.checkbox("Mostrar barras de error en evaluación", value=True)

    # ==============================================================
    # Feature Engineering
    # ==============================================================

    df["Mes"] = df.index.month
    df["Año"] = df.index.year
    df["sin_mes"] = np.sin(2 * np.pi * df["Mes"]/12)
    df["cos_mes"] = np.cos(2 * np.pi * df["Mes"]/12)

    for c in exog_cols + [col_obj]:
        for lag in range(1, 13):
            df[f"{c}_lag{lag}"] = df[c].shift(lag)

    df.dropna(inplace=True)

    y = df[col_obj]
    X = df.drop(columns=[col_obj])

    test_size = int(0.1 * len(df))
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    scalerX = RobustScaler()
    scalerY = RobustScaler()
    X_train_s = scalerX.fit_transform(X_train)
    X_test_s = scalerX.transform(X_test)
    y_train_s = scalerY.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    # ==============================================================
    # Modelos
    # ==============================================================

    modelos = {
        "XGBoost": XGBRegressor(
            n_estimators=1500, learning_rate=0.02, max_depth=7,
            subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.2, reg_lambda=1.0, random_state=42
        ),
        "Random Forest": RandomForestRegressor(n_estimators=1200, max_depth=12, random_state=42),
        "SVM": SVR(kernel="rbf", C=15, epsilon=0.05, gamma=0.05)
    }

    resultados = {}
    for name, model in modelos.items():
        model.fit(X_train_s, y_train_s)
        y_pred_s = model.predict(X_test_s)
        y_pred = scalerY.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
        resultados[name] = y_pred

    # ==============================================================
    # Selección del mejor modelo
    # ==============================================================

    mape_scores = {name: mape(y_test, pred) for name, pred in resultados.items()}
    best_model = min(mape_scores, key=mape_scores.get)
    best_mape = mape_scores[best_model]

    st.success(f"🏆 Mejor modelo: **{best_model}** con MAPE = {best_mape:.2f}%")

    # ==============================================================
    # Pronóstico 12 meses
    # ==============================================================

    horizon = 12
    fechas_futuras = pd.date_range(df.index[-1] + timedelta(days=30), periods=horizon, freq="M")
    ultima_fila = X.iloc[-1:].copy()
    X_future = pd.concat([ultima_fila] * horizon, ignore_index=True)
    X_future.index = fechas_futuras
    X_future_s = scalerX.transform(X_future)

    y_future_s = modelos[best_model].predict(X_future_s)
    y_future = scalerY.inverse_transform(y_future_s.reshape(-1, 1)).ravel()

    # ==============================================================
    # 📊 GRAFICO COMPLETO CON PERIODOS Y OPCIÓN DE BARRAS DE ERROR
    # ==============================================================

    st.subheader("📈 Serie completa: Entrenamiento, Evaluación y Pronóstico")

    serie_train = pd.Series(y_train.values, index=y_train.index, name="Entrenamiento")
    serie_test_real = pd.Series(y_test.values, index=y_test.index, name="Evaluación Real")
    serie_test_pred = pd.Series(resultados[best_model], index=y_test.index, name=f"Evaluación Predicha ({best_model})")
    serie_forecast = pd.Series(y_future, index=fechas_futuras, name="Pronóstico Futuro (12M)")

    errores = np.abs(serie_test_real - serie_test_pred)

    fig = go.Figure()

    # Entrenamiento
    fig.add_trace(go.Scatter(
        x=serie_train.index, y=serie_train.values,
        mode="lines", name="Entrenamiento (Real)",
        line=dict(color="black", width=2)
    ))

    # Evaluación real
    fig.add_trace(go.Scatter(
        x=serie_test_real.index, y=serie_test_real.values,
        mode="lines", name="Evaluación (Real)",
        line=dict(color="gray", width=2)
    ))

    # Evaluación predicha (punteada + opcional error bars)
    error_config = dict(type="data", array=errores, visible=show_errors, color="rgba(255,140,0,0.4)")
    fig.add_trace(go.Scatter(
        x=serie_test_pred.index, y=serie_test_pred.values,
        error_y=error_config,
        mode="lines+markers", name=f"Ajuste del Modelo ({best_model})",
        line=dict(color="orange", width=3, dash="dot"),
        marker=dict(size=5, color="orange")
    ))

    # Pronóstico futuro (línea punteada verde)
    fig.add_trace(go.Scatter(
        x=serie_forecast.index, y=serie_forecast.values,
        mode="lines+markers", name="Pronóstico 12M",
        line=dict(color="green", dash="dot", width=3),
        marker=dict(size=5, color="green")
    ))

    # ==============================================================
    # Líneas verticales (usando shapes y anotaciones seguras)
    # ==============================================================

    y_min, y_max = min(y.min(), y_future.min()), max(y.max(), y_future.max())

    fig.add_shape(
        type="line", x0=y_train.index[-1], x1=y_train.index[-1],
        y0=y_min, y1=y_max,
        line=dict(color="black", width=2, dash="dash")
    )
    fig.add_annotation(
        x=y_train.index[-1], y=y_max, text="Fin Entrenamiento",
        showarrow=False, yshift=10, font=dict(color="black")
    )

    fig.add_shape(
        type="line", x0=y_test.index[-1], x1=y_test.index[-1],
        y0=y_min, y1=y_max,
        line=dict(color="gray", width=2, dash="dash")
    )
    fig.add_annotation(
        x=y_test.index[-1], y=y_max, text="Fin Evaluación",
        showarrow=False, yshift=10, font=dict(color="gray")
    )

    fig.update_layout(
        template="plotly_white",
        title=f"Evolución del Precio de la Soya – {best_model}: Entrenamiento, Evaluación y Pronóstico",
        xaxis_title="Fecha",
        yaxis_title="Precio (USD/TM)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ==============================================================
    # 📋 TABLA DE COMPARACIÓN REAL VS PREDICHO
    # ==============================================================

    st.subheader("📋 Comparación: Real vs Predicho en Muestra de Evaluación")

    df_eval = pd.DataFrame({
        "Fecha": y_test.index,
        "Real": y_test.values,
        f"Predicho ({best_model})": resultados[best_model],
    })
    df_eval["Error Abs."] = np.abs(df_eval["Real"] - df_eval[f"Predicho ({best_model})"])
    df_eval["Error (%)"] = np.abs((df_eval["Real"] - df_eval[f"Predicho ({best_model})"]) / df_eval["Real"]) * 100

    st.dataframe(df_eval.style.format({"Real": "{:.2f}", f"Predicho ({best_model})": "{:.2f}", "Error Abs.": "{:.2f}", "Error (%)": "{:.2f}"}))
    st.caption("📊 Valores reales vs predichos por el modelo ganador, con error absoluto y porcentual.")

    # ==============================================================
    # Descarga Excel
    # ==============================================================

    df_pred = pd.DataFrame({
        "Fecha": fechas_futuras,
        "Pronóstico": y_future
    })
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_pred.to_excel(writer, index=False, sheet_name="Pronóstico")
    st.download_button("📥 Descargar Pronóstico (Excel)", data=buffer.getvalue(),
                       file_name="Pronostico_Soya_2025_2026.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
