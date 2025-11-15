# ==============================================================
# 🧠 SolverTic SoyAI Predictor – Sistema Inteligente de Modelado del Precio de la Soya
# Versión 5.6 Modern Pro Extended – con Ensemble Manual y Pronóstico Futuro
# ==============================================================

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import streamlit as st
st.set_page_config(page_title="SolverTic SoyAI Predictor", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

# Modelado estadístico
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from prophet import Prophet
import plotly.graph_objects as go
import math
from sklearn.model_selection import TimeSeriesSplit
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

def select_differencing(y):
    try:
        return 1 if adfuller(y.dropna())[1] > 0.05 else 0
    except Exception:
        return 0

# ==============================================================
# FUNCIONES SARIMAX
# ==============================================================

def fit_model(y, order, seasonal_order, exog=None):
    try:
        model = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                        exog=exog, enforce_stationarity=False,
                        enforce_invertibility=False)
        return model.fit(disp=False)
    except Exception:
        return None

def diagnosticos(res):
    resid = res.resid.dropna()
    try:
        jb_p = jarque_bera(resid)[1]
        lb_p = acorr_ljungbox(resid, lags=[min(24, len(resid)//2)],
                              return_df=True)["lb_pvalue"].iloc[0]
        arch_p = het_arch(resid, nlags=12)[1]
    except Exception:
        jb_p, lb_p, arch_p = 1, 1, 1
    return jb_p, lb_p, arch_p, resid

def buscar_modelos(train, test, pmax, qmax, Pmax, Qmax, periodo):
    st.info("🔍 Buscando el mejor modelo SARIMAX... puede tardar unos segundos.")
    results = []
    d = select_differencing(train)
    total = (pmax+1)*(qmax+1)*(Pmax+1)*(Qmax+1)
    bar = st.progress(0)
    combos = [(p, q, P, Q) for p in range(pmax+1)
              for q in range(qmax+1)
              for P in range(Pmax+1)
              for Q in range(Qmax+1)]
    for i, (p, q, P, Q) in enumerate(combos):
        bar.progress(int((i+1)/total*100))
        order = (p, d, q)
        seasonal_order = (P, 1, Q, periodo)
        try:
            res = fit_model(train, order, seasonal_order)
            if res is None:
                continue
            fc = res.get_forecast(steps=len(test)).predicted_mean
            jb_p, lb_p, arch_p, resid = diagnosticos(res)
            results.append({
                'order': order, 'seasonal': seasonal_order,
                'aic': res.aic, 'mape': mape(test, fc),
                'valid': (jb_p > 0.05) & (lb_p > 0.05) & (arch_p > 0.05),
                'res': res, 'forecast': fc
            })
        except Exception:
            continue
    if not results:
        st.warning("⚠️ No se encontraron modelos válidos.")
        return None, None
    df = pd.DataFrame(results)
    best = df.sort_values(['valid', 'mape', 'aic'], ascending=[False, True, True]).iloc[0]
    return df, best

# ==============================================================
# INTERFAZ CON TABS
# ==============================================================

tab1, tab2 = st.tabs(["📊 SARIMAX Tradicional", "🤖 Machine Learning Avanzado"])

# ==============================================================
# TAB 1 – SARIMAX
# ==============================================================

with tab1:
    st.title("📊 Modelado SARIMAX Tradicional")
    with st.sidebar:
        st.header("Configuración SARIMAX")
        file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
        pmax = st.slider("p/q máximo", 1, 5, 2)
        Pmax = st.slider("P/Q máximo (estacional)", 0, 3, 1)
        periodo_estacional = st.number_input("Periodo estacional (meses)", 3, 24, 12)
        test_size = st.slider("Meses para prueba", 6, 36, 24)
    if file:
        df = pd.read_csv(file)
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0]).sort_index()
        serie = limpiar_serie(df.iloc[:, 0])
        train, test = serie[:-test_size], serie[-test_size:]
        df_res, best = buscar_modelos(train, test, pmax, pmax, Pmax, Pmax, periodo_estacional)
        if best is not None:
            st.success("✅ Modelado completado")
            st.metric("Mejor MAPE", f"{best['mape']:.2f}%")
            st.metric("AIC", f"{best['aic']:.2f}")
            fig, ax = plt.subplots(figsize=(8, 4))
            train.plot(ax=ax, label="Train")
            test.plot(ax=ax, label="Test")
            best["forecast"].plot(ax=ax, label="Pronóstico", color="red")
            ax.legend()
            st.pyplot(fig)

# ==============================================================
# TAB 2 – MACHINE LEARNING AVANZADO (CON ENSEMBLE + FUTURE FORECAST)
# ==============================================================

with tab2:
    st.title("🤖 Machine Learning Avanzado – SolverTic SoyAI Predictor")
    st.markdown("### Modelos: XGBoost, Random Forest, SVM (Optimizado), MLP y Prophet 🌱")

    file_ml = st.file_uploader("📂 Sube tu archivo CSV con variables", type=["csv"], key="ml_file")
    if file_ml:
        df = pd.read_csv(file_ml)
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0]).sort_index()

        col_obj = st.selectbox("🎯 Variable objetivo", df.columns)
        exog_cols = [c for c in df.columns if c != col_obj]

        # Crear variables temporales y lags
        df["Mes"] = df.index.month
        df["Año"] = df.index.year
        df["Trimestre"] = df.index.quarter
        df["sin_mes"] = np.sin(2 * np.pi * df["Mes"]/12)
        df["cos_mes"] = np.cos(2 * np.pi * df["Mes"]/12)
        for c in exog_cols:
            df[f"{c}_lag1"] = df[c].shift(1)
        df.dropna(inplace=True)

        y = df[col_obj]
        X = df.drop(columns=[col_obj])

        # División 90/10
        test_size = int(0.1 * len(df))
        X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
        y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

        # Escalado
        scalerX = StandardScaler()
        scalerY = StandardScaler()
        X_train_s = scalerX.fit_transform(X_train)
        X_test_s = scalerX.transform(X_test)
        y_train_s = scalerY.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        # MODELOS OPTIMIZADOS
        modelos = {
            "XGBoost": XGBRegressor(
                n_estimators=800, learning_rate=0.03, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42),
            "Random Forest": RandomForestRegressor(
                n_estimators=1000, max_depth=10, min_samples_split=4,
                min_samples_leaf=2, random_state=42),
            "SVM (Optimizado)": SVR(kernel="rbf", C=50, epsilon=0.02, gamma=0.07),
            "Neural Network": MLPRegressor(hidden_layer_sizes=(60, 60),
                                           max_iter=2000, random_state=42)
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

        # VALIDACIÓN CRUZADA TEMPORAL (SVM)
        tscv = TimeSeriesSplit(n_splits=5)
        svr = SVR(kernel="rbf", C=50, epsilon=0.02, gamma=0.07)
        scores = []
        for train_idx, test_idx in tscv.split(X_train_s):
            svr.fit(X_train_s[train_idx], y_train_s[train_idx])
            y_pred_val = svr.predict(X_train_s[test_idx])
            y_pred_inv = scalerY.inverse_transform(y_pred_val.reshape(-1, 1)).ravel()
            y_true_inv = scalerY.inverse_transform(y_train_s[test_idx].reshape(-1, 1)).ravel()
            scores.append(mape(y_true_inv, y_pred_inv))
        mape_cv = np.mean(scores)
        st.success(f"✅ MAPE promedio validación cruzada (SVM Optimizado): {mape_cv:.2f}%")

        # MÉTRICAS
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
        st.subheader("📋 Resultados de Modelos Optimizados")
        st.dataframe(df_metrics.style.highlight_min(subset=["MAPE"], color="lightgreen"))

        # ENSEMBLE MANUAL
        st.subheader("🎛️ Ensemble Manual (ajusta pesos de modelos)")

        w_svm = st.slider("Peso SVM (Optimizado)", 0.0, 1.0, 0.5)
        w_xgb = st.slider("Peso XGBoost", 0.0, 1.0, 0.3)
        w_rf = st.slider("Peso Random Forest", 0.0, 1.0, 0.2)

        suma = w_svm + w_xgb + w_rf
        if suma == 0: suma = 1

        y_ensemble = (w_svm * resultados["SVM (Optimizado)"] +
                      w_xgb * resultados["XGBoost"] +
                      w_rf * resultados["Random Forest"]) / suma

        st.success(f"📉 MAPE Ensemble Manual: {mape(y_test, y_ensemble):.2f}%")

        # GRÁFICO COMPARATIVO
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test.index, y=y_test, name="Real", line=dict(color="black")))
        for name, y_pred in resultados.items():
            fig.add_trace(go.Scatter(x=y_test.index, y=y_pred, name=name))
        fig.add_trace(go.Scatter(x=y_test.index, y=y_ensemble,
                                 name="Ensemble Manual", line=dict(color="green", width=3)))
        fig.update_layout(title="Comparación de Predicciones – Modelos + Ensemble",
                          xaxis_title="Fecha", yaxis_title="Precio (USD/TM)",
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # ==============================================================
        # 🔮 PRONÓSTICO FUTURO 12 MESES
        # ==============================================================

        st.subheader("🔮 Pronóstico Futuro – Junio 2025 a Mayo 2026")

        horizon = 12
        fechas_futuras = pd.date_range(df.index[-1] + timedelta(days=30), periods=horizon, freq="M")

        # Pronóstico SVM
        X_future = X.iloc[-1:].copy()
        for i in range(horizon):
            X_future = pd.concat([X_future, X_future.tail(1)], ignore_index=True)
        X_future.index = fechas_futuras

        X_future_s = scalerX.transform(X_future)
        y_future_svm_s = modelos["SVM (Optimizado)"].predict(X_future_s)
        y_future_svm = scalerY.inverse_transform(y_future_svm_s.reshape(-1, 1)).ravel()

        # Pronóstico Ensemble
        y_future_ensemble = (w_svm * y_future_svm +
                             w_xgb * resultados["XGBoost"][-horizon:] +
                             w_rf * resultados["Random Forest"][-horizon:]) / (w_svm + w_xgb + w_rf)

        # Visualización
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=df.index, y=y, name="Histórico", line=dict(color="#2E8B57")))
        fig_future.add_trace(go.Scatter(x=fechas_futuras, y=y_future_svm, name="Pronóstico SVM (Optimizado)",
                                        line=dict(color="red", width=2, dash="dot")))
        fig_future.add_trace(go.Scatter(x=fechas_futuras, y=y_future_ensemble, name="Pronóstico Ensemble Manual",
                                        line=dict(color="green", width=3)))
        fig_future.update_layout(title="🔮 Pronóstico 12 Meses – Modelos Avanzados",
                                 xaxis_title="Fecha", yaxis_title="Precio proyectado (USD/TM)",
                                 template="plotly_white")
        st.plotly_chart(fig_future, use_container_width=True)

        # Tabla resumen
        df_pred = pd.DataFrame({
            "Fecha": fechas_futuras,
            "Pronóstico_SVM_Optimizado": y_future_svm,
            "Pronóstico_Ensemble_Manual": y_future_ensemble
        })
        st.subheader("📈 Valores proyectados (Jun 2025 – May 2026)")
        st.dataframe(df_pred.style.format({"Pronóstico_SVM_Optimizado": "{:.2f}",
                                           "Pronóstico_Ensemble_Manual": "{:.2f}"}))

        # ==============================================================
        # 💾 DESCARGA DEL PRONÓSTICO
        # ==============================================================
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_pred.to_excel(writer, index=False, sheet_name='Pronóstico')
        st.download_button(
            label="📥 Descargar Pronóstico (Excel)",
            data=buffer.getvalue(),
            file_name="Pronostico_Soya_2025_2026.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
