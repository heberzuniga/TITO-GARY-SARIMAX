# ==============================================================
# 🧠 Sistema Inteligente de Modelado del Precio de la Soya – SolverTic SRL
# Versión 5.3 FINAL – SARIMAX + ML Avanzado + Sidebar restaurado
# ==============================================================

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import streamlit as st
st.set_page_config(page_title="Sistema Inteligente de Modelado del Precio de la Soya", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Librerías para ML
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from prophet import Prophet
from datetime import timedelta
import math

# ==============================================================
# FUNCIONES AUXILIARES
# ==============================================================

def winsorize_series(s, low_q=0.01, high_q=0.99):
    lo, hi = s.quantile(low_q), s.quantile(high_q)
    return s.clip(lower=lo, upper=hi)

def limpiar_serie(s, winsor=True):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if winsor:
        s = winsorize_series(s)
    return s.interpolate(method="linear").bfill().ffill()

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def theil_u1(y_true, y_pred):
    num = np.sqrt(np.mean((y_pred - y_true)**2))
    den = np.sqrt(np.mean(y_true**2)) + np.sqrt(np.mean(y_pred**2))
    return num / den

def theil_u2(y_true, y_pred):
    return np.sqrt(np.sum((y_pred - y_true)**2) / np.sum((y_true[1:] - y_true[:-1])**2))

# ==============================================================
# FUNCIONES SARIMAX
# ==============================================================

def fit_model(y, order, seasonal_order, exog=None):
    try:
        model = SARIMAX(y, order=order, seasonal_order=seasonal_order, exog=exog,
                        enforce_stationarity=False, enforce_invertibility=False)
        return model.fit(disp=False)
    except Exception:
        return None

def diagnosticos(res):
    resid = res.resid.dropna()
    try:
        jb_p = jarque_bera(resid)[1]
        lb_p = acorr_ljungbox(resid, lags=[min(24, len(resid)//2)], return_df=True)["lb_pvalue"].iloc[0]
        arch_p = het_arch(resid, nlags=12)[1]
    except Exception:
        jb_p, lb_p, arch_p = 1, 1, 1
    return jb_p, lb_p, arch_p, resid

def select_differencing(y):
    try:
        return 1 if adfuller(y.dropna())[1] > 0.05 else 0
    except Exception:
        return 0

def fourier_terms(index, period=12, K=1):
    t = np.arange(len(index))
    data = {}
    for k in range(1, K + 1):
        data[f'sin_{k}'] = np.sin(2 * np.pi * k * t / period)
        data[f'cos_{k}'] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(data, index=index)

def buscar_modelos(train, test, pmax, qmax, Pmax, Qmax, periodo, include_fourier, K_min, K_max):
    st.info("🔍 Buscando el mejor modelo... esto puede tardar unos segundos.")
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
            if include_fourier:
                for K in range(K_min, K_max+1):
                    Xtrain = fourier_terms(train.index, periodo, K)
                    Xtest = fourier_terms(test.index, periodo, K)
                    res = fit_model(train, order, seasonal_order, exog=Xtrain)
                    if res is None:
                        continue
                    fc = res.get_forecast(steps=len(test), exog=Xtest).predicted_mean
                    jb_p, lb_p, arch_p, resid = diagnosticos(res)
                    results.append({
                        'order': order, 'seasonal': seasonal_order,
                        'fourier_K': K, 'aic': res.aic, 'mape': mape(test, fc),
                        'jb_p': jb_p, 'lb_p': lb_p, 'arch_p': arch_p,
                        'valid': (jb_p > 0.05) & (lb_p > 0.05) & (arch_p > 0.05),
                        'res': res, 'forecast': fc, 'resid': resid})
            else:
                res = fit_model(train, order, seasonal_order)
                if res is None:
                    continue
                fc = res.get_forecast(steps=len(test)).predicted_mean
                jb_p, lb_p, arch_p, resid = diagnosticos(res)
                results.append({
                    'order': order, 'seasonal': seasonal_order,
                    'fourier_K': None, 'aic': res.aic, 'mape': mape(test, fc),
                    'jb_p': jb_p, 'lb_p': lb_p, 'arch_p': arch_p,
                    'valid': (jb_p > 0.05) & (lb_p > 0.05) & (arch_p > 0.05),
                    'res': res, 'forecast': fc, 'resid': resid})
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

tab1, tab2 = st.tabs(["📊 Modelado SARIMAX Tradicional", "🤖 Machine Learning Avanzado"])

# ==============================================================
# TAB 1 – SARIMAX (con sidebar restaurado)
# ==============================================================

with tab1:
    st.title("🧠 Sistema Inteligente de Modelado del Precio de la Soya")
    st.caption("SolverTic SRL – División de Inteligencia Artificial y Modelado Predictivo")

    with st.sidebar:
        st.header("📂 Cargar y Configurar")
        file = st.file_uploader("Sube tu archivo CSV de precios mensuales", type=['csv'])
        pmax = st.slider("Máx p/q", 1, 5, 3)
        Pmax = st.slider("Máx P/Q (estacional)", 0, 3, 1)
        include_fourier = st.checkbox("Incluir Fourier (SARIMAX)", value=True)
        K_min, K_max = st.slider("Rango K Fourier", 1, 6, (1, 3))
        periodo_estacional = st.number_input("Periodo estacional (meses)", 3, 24, 12)
        test_size = st.slider("Meses para Test", 6, 36, 24)
        fecha_inicio = st.date_input("Inicio de análisis", datetime.date(2010, 1, 1))
        fecha_fin = st.date_input("Fin de análisis", datetime.date(2025, 5, 31))
        winsor = st.checkbox("Capar outliers (winsorizar)", value=True)
        st.caption("© 2025 SolverTic SRL – Ingeniería de Sistemas Inteligentes")

    if file:
        df = pd.read_csv(file)
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0]).sort_index()
        serie = limpiar_serie(df.iloc[:, 0], winsor=winsor)
        serie = serie.loc[(serie.index >= str(fecha_inicio)) & (serie.index <= str(fecha_fin))]

        if len(serie) <= test_size:
            test_size = max(1, len(serie)//5)
        train = serie[:-test_size]
        test = serie[-test_size:]

        st.subheader("📈 Vista previa de datos")
        st.line_chart(serie)
        st.write(f"**Observaciones:** {len(serie)} | Train={len(train)} | Test={len(test)}")

        df_res, best = buscar_modelos(train, test, pmax, qmax=pmax,
                                      Pmax=Pmax, Qmax=Pmax,
                                      periodo=periodo_estacional,
                                      include_fourier=include_fourier,
                                      K_min=K_min, K_max=K_max)

        if df_res is not None:
            st.success("✅ Modelado completado exitosamente")
            c1, c2, c3 = st.columns(3)
            c1.metric("Mejor MAPE", f"{best['mape']:05.2f}%")
            c2.metric("AIC", f"{best['aic']:05.2f}")
            c3.metric("Modelos válidos", f"{df_res['valid'].sum()}/{len(df_res)}")

            st.subheader("🏆 Top 10 modelos por MAPE")
            tabla = df_res.sort_values('mape').head(10)[['order', 'seasonal', 'fourier_K', 'mape', 'aic']].copy()
            tabla['mape'] = tabla['mape'].map(lambda x: f"{x:05.2f}")
            tabla['aic'] = tabla['aic'].map(lambda x: f"{x:05.2f}")
            st.dataframe(tabla)

            fig, ax = plt.subplots()
            ax.scatter(df_res['aic'], df_res['mape'], alpha=0.7, color='seagreen')
            ax.set_xlabel('AIC'); ax.set_ylabel('MAPE (%)')
            ax.set_title('Relación AIC vs MAPE')
            st.pyplot(fig)

            res_best = best['res']
            fc = best['forecast']
            resid_best = best['resid']
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            train.plot(ax=ax2, label='Train')
            test.plot(ax=ax2, label='Test')
            fc.plot(ax=ax2, label='Pronóstico', color='red')
            ax2.legend(); st.pyplot(fig2)

# ==============================================================
# TAB 2 – MACHINE LEARNING AVANZADO (sin sidebar)
# ==============================================================

with tab2:
    st.header("🤖 Modelos Avanzados de Machine Learning – SolverTic SRL")
    st.markdown("""
    Compara **XGBoost, Random Forest, SVM, Redes Neuronales (MLP)** y **Prophet**  
    para pronóstico del precio de la soya (2009–2025).  
    El modelo con menor **MAPE** será el mejor.
    """)

    file_ml = st.file_uploader("📂 Subir archivo CSV con variables", type=["csv"], key="ml_file")
    if file_ml:
        df = pd.read_csv(file_ml)
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0]).sort_index()

        col_obj = st.selectbox("🎯 Variable objetivo", df.columns)
        exog_cols = st.multiselect("📈 Variables exógenas (opcionales)", [c for c in df.columns if c != col_obj])

        if st.button("🚀 Ejecutar Modelado ML"):
            y = df[col_obj].astype(float)
            X = df[exog_cols] if exog_cols else pd.DataFrame(np.arange(len(y)), columns=["indice"])

            test_size = int(0.2 * len(df))
            X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
            y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

            modelos = {
                "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5),
                "Random Forest": RandomForestRegressor(n_estimators=400, random_state=42),
                "SVM": SVR(kernel='rbf', C=10, epsilon=0.1),
                "Neural Network": MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=2000, random_state=42)
            }

            resultados = {m: modelos[m].fit(X_train, y_train).predict(X_test) for m in modelos}

            df_prophet = pd.DataFrame({"ds": df.index, "y": y.values})
            model_prophet = Prophet(yearly_seasonality=True)
            model_prophet.fit(df_prophet.iloc[:-test_size])
            future = model_prophet.make_future_dataframe(periods=test_size, freq="M")
            resultados["Prophet"] = model_prophet.predict(future)["yhat"].iloc[-test_size:].values

            metrics = [[m, math.sqrt(mean_squared_error(y_test, y_p)), mean_absolute_error(y_test, y_p),
                        mape(y_test, y_p), smape(y_test, y_p), theil_u1(y_test, y_p), theil_u2(y_test, y_p)]
                       for m, y_p in resultados.items()]

            df_metrics = pd.DataFrame(metrics, columns=["Modelo", "RMSE", "MAE", "MAPE", "SMAPE", "Theil U1", "Theil U2"])
            best_model = df_metrics.loc[df_metrics["MAPE"].idxmin(), "Modelo"]

            st.dataframe(df_metrics)
            st.success(f"🏆 El mejor modelo es **{best_model}**.")

            # Pronóstico futuro
            horizon = 12
            fechas_futuras = pd.date_range(df.index[-1] + timedelta(days=30), periods=horizon, freq="M")

            if best_model != "Prophet":
                model_best = modelos[best_model]
                X_future = (pd.DataFrame(np.tile(X.iloc[-1].values, (horizon, 1)), columns=X.columns)
                            if exog_cols else pd.DataFrame(np.arange(len(df), len(df) + horizon), columns=X.columns))
                pred_future = model_best.predict(X_future)
            else:
                future2 = model_prophet.make_future_dataframe(periods=horizon, freq="M")
                pred_future = model_prophet.predict(future2)["yhat"].iloc[-horizon:].values

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df.index, y=y, name="Histórico", line=dict(color="#2E8B57")))
            fig2.add_trace(go.Scatter(x=fechas_futuras, y=pred_future, name=f"Pronóstico {best_model}",
                                      line=dict(color="red", width=3)))
            fig2.update_layout(title="Pronóstico a 12 meses",
                               xaxis_title="Fecha",
                               yaxis_title="Precio proyectado (USD/TM)",
                               template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)
