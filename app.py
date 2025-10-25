# ==============================================================
# 🧠 Sistema Inteligente de Modelado del Precio de la Soya – SolverTic SRL
# Versión 4.6 FINAL – MAPE y AIC con formato (02.02)
# ==============================================================

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"  # evita errores de matplotlib en Streamlit Cloud

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

# ==============================================================
# BÚSQUEDA DE MODELOS
# ==============================================================

def buscar_modelos(train, test, pmax, qmax, Pmax, Qmax,
                   periodo, include_fourier, K_min, K_max):
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
    best = df.sort_values(['valid', 'mape', 'aic'],
                          ascending=[False, True, True]).iloc[0]
    return df, best

# ==============================================================
# INTERFAZ PRINCIPAL
# ==============================================================

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
        tabla = df_res.sort_values('mape').head(10)[
            ['order', 'seasonal', 'fourier_K', 'mape', 'aic']
        ].copy()
        tabla['mape'] = tabla['mape'].map(lambda x: f"{x:05.2f}")
        tabla['aic'] = tabla['aic'].map(lambda x: f"{x:05.2f}")
        st.dataframe(tabla)

        # Gráfico AIC vs MAPE
        fig, ax = plt.subplots()
        ax.scatter(df_res['aic'], df_res['mape'], alpha=0.7, color='seagreen')
        ax.set_xlabel('AIC')
        ax.set_ylabel('MAPE (%)')
        ax.set_title('Relación AIC vs MAPE')
        st.pyplot(fig)

        # Pronóstico
        res_best = best['res']
        fc = best['forecast']
        resid_best = best['resid']
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        train.plot(ax=ax2, label='Train')
        test.plot(ax=ax2, label='Test')
        fc.plot(ax=ax2, label='Pronóstico', color='red')
        ax2.legend()
        st.pyplot(fig2)

        # Residuales
        fig_r, ax_r = plt.subplots(figsize=(8, 3))
        resid_best.plot(ax=ax_r)
        ax_r.set_title("Residuales en el tiempo")
        st.pyplot(fig_r)

        # Histograma de residuales
        fig_h, ax_h = plt.subplots(figsize=(6, 3))
        ax_h.hist(resid_best, bins=20, color='skyblue', edgecolor='black', alpha=0.7, density=True)
        x = np.linspace(ax_h.get_xlim()[0], ax_h.get_xlim()[1], 100)
        ax_h.plot(x, stats.norm.pdf(x, resid_best.mean(), resid_best.std()), 'r', linewidth=2)
        ax_h.set_title("Histograma de residuales con curva normal")
        st.pyplot(fig_h)

        # Q-Q Plot
        fig_qq = plt.figure(figsize=(5, 3))
        stats.probplot(resid_best, dist="norm", plot=plt)
        plt.title("Q–Q Plot de los residuales")
        st.pyplot(fig_qq)

        # ACF y PACF
        fig_acf = plt.figure(figsize=(5, 3))
        plot_acf(resid_best, lags=min(24, len(resid_best)//2), ax=plt.gca())
        plt.title("ACF de los residuales")
        st.pyplot(fig_acf)

        fig_pacf = plt.figure(figsize=(5, 3))
        plot_pacf(resid_best, lags=min(24, len(resid_best)//2), ax=plt.gca(), method='ywm')
        plt.title("PACF de los residuales")
        st.pyplot(fig_pacf)
else:
    st.info("👆 Sube un archivo CSV con tu serie mensual para comenzar el modelado.")
