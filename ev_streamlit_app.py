"""
EV Stock Forecast Streamlit App
Models: Reference (sim) | Auto-ARIMA | Prophet | STS
- Log-normal prediction intervals (ARIMA / Prophet / STS)
- Backtest evaluation (MAE / MAPE / RMSE)
- 4-scenario analysis with exogenous variables
- Auto-ARIMA Baseline overlay for comparison
- Interactive Plotly charts + CSV export
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Optional model backends ───────────────────────────────────────────────────
try:
    from pmdarima import auto_arima
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

try:
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    HAS_STS = True
except ImportError:
    HAS_STS = False

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════
YEARS_OBS = [2019, 2020, 2021, 2022, 2023, 2024]
YEARS_FC  = [2025, 2026, 2027, 2028, 2029, 2030]
N_FC      = len(YEARS_FC)

EV_DEFAULT = [100, 500, 1500, 3500, 5000, 5000]

EXOG_VARS = [
    "Subsidy (10k JPY/unit)",
    "Charging Stations",
    "Gas Price (JPY/L)",
    "EV Model Count",
    "GDP Growth Rate (%)",
]

EXOG_OBS: dict[str, list] = {
    "Subsidy (10k JPY/unit)":  [0,    10,   50,    100,   80,    60   ],
    "Charging Stations":       [1000, 2000, 5000,  10000, 15000, 20000],
    "Gas Price (JPY/L)":       [150,  130,  145,   170,   175,   165  ],
    "EV Model Count":          [3,    5,    8,     15,    20,    25   ],
    "GDP Growth Rate (%)":     [0.5, -4.1,  2.1,   1.0,   1.9,   0.5  ],
}

SCENARIOS: dict[str, dict[str, list]] = {
    "Baseline": {
        "Subsidy (10k JPY/unit)":  [55,    50,    45,    40,    35,    30   ],
        "Charging Stations":       [25000, 30000, 35000, 40000, 45000, 50000],
        "Gas Price (JPY/L)":       [165,   167,   168,   170,   170,   170  ],
        "EV Model Count":          [28,    32,    36,    40,    44,    48   ],
        "GDP Growth Rate (%)":     [1.0,   1.0,   1.0,   1.0,   1.0,   1.0  ],
    },
    "Policy Boost": {
        "Subsidy (10k JPY/unit)":  [70,    75,    80,    80,    80,    80   ],
        "Charging Stations":       [30000, 40000, 52000, 62000, 72000, 80000],
        "Gas Price (JPY/L)":       [175,   180,   185,   190,   195,   200  ],
        "EV Model Count":          [32,    38,    44,    50,    56,    60   ],
        "GDP Growth Rate (%)":     [1.5,   1.5,   1.5,   1.5,   1.5,   1.5  ],
    },
    "Infra Constraint": {
        "Subsidy (10k JPY/unit)":  [50,    45,    35,    25,    20,    20   ],
        "Charging Stations":       [21000, 22500, 24000, 25500, 27000, 28000],
        "Gas Price (JPY/L)":       [160,   158,   157,   155,   155,   155  ],
        "EV Model Count":          [26,    28,    31,    34,    37,    40   ],
        "GDP Growth Rate (%)":     [0.5,   0.5,   0.5,   0.5,   0.5,   0.5  ],
    },
    "Price Drop": {
        "Subsidy (10k JPY/unit)":  [50,    40,    30,    20,    20,    20   ],
        "Charging Stations":       [26000, 32000, 38000, 46000, 52000, 58000],
        "Gas Price (JPY/L)":       [180,   190,   200,   210,   215,   220  ],
        "EV Model Count":          [35,    44,    52,    60,    65,    70   ],
        "GDP Growth Rate (%)":     [1.2,   1.2,   1.2,   1.2,   1.2,   1.2  ],
    },
}

SC_COLORS = {
    "Baseline":         "deepskyblue",
    "Policy Boost":     "green",
    "Infra Constraint": "orange",
    "Price Drop":       "purple",
}

EXOG_BOUNDS = {
    "Subsidy (10k JPY/unit)":  (0.0,   120.0,   1.0),
    "Charging Stations":       (1000., 150000., 1000.),
    "Gas Price (JPY/L)":       (80.0,  400.0,   5.0),
    "EV Model Count":          (1.0,   150.0,   1.0),
    "GDP Growth Rate (%)":     (-5.0,  15.0,    0.1),
}

# Reference baseline — mirrors ev_forecast_simulation.py exactly
BASELINE_TREND_STEP = 4000
BASELINE_C          = 729.0

# Model options (filter to installed backends)
_ALL_MODELS = ["Auto-ARIMA", "Prophet", "STS (Structural)"]
AVAILABLE_MODELS = (
    (["Auto-ARIMA"] if HAS_PMDARIMA else [])
    + (["Prophet"]         if HAS_PROPHET  else [])
    + (["STS (Structural)"] if HAS_STS      else [])
    + ([] if (HAS_PMDARIMA or HAS_PROPHET or HAS_STS) else ["Fallback (log-linear)"])
)
if not AVAILABLE_MODELS:
    AVAILABLE_MODELS = ["Fallback (log-linear)"]

# ══════════════════════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════════════════════

def z_normalize(arr, mu=None, sigma=None):
    arr = np.asarray(arr, float)
    if mu is None:
        mu    = arr.mean(axis=0)
        sigma = arr.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (arr - mu) / sigma, mu, sigma


def build_exog(var_dict: dict) -> np.ndarray:
    return np.column_stack([var_dict[k] for k in EXOG_VARS])


def to_t(arr: np.ndarray) -> tuple:
    return tuple(map(tuple, arr.tolist()))


# ══════════════════════════════════════════════════════════════════════════════
# Model implementations
# ══════════════════════════════════════════════════════════════════════════════

# ── Auto-ARIMA ────────────────────────────────────────────────────────────────

def _fit_arima(ev_raw, exog_obs_s, exog_fc_s, use_exog, n_periods):
    """
    Auto-ARIMA with d=1 on ORIGINAL scale.
    Forcing d=1 (first-difference) prevents explosive/collapsing forecasts
    that occur on log scale with only 6 data points.
    Intervals are symmetric; non-negative via np.maximum(0, ...).
    """
    X_obs = exog_obs_s if use_exog else None
    X_fc  = exog_fc_s  if use_exog else None

    if HAS_PMDARIMA:
        model = auto_arima(
            ev_raw, X=X_obs, d=1,          # force first-difference for stability
            seasonal=False, stepwise=True,
            information_criterion="aic", max_p=2, max_q=2,
            error_action="ignore", suppress_warnings=True,
        )
        fc, ci95 = model.predict(n_periods, X=X_fc, return_conf_int=True, alpha=0.05)
        _, ci50  = model.predict(n_periods, X=X_fc, return_conf_int=True, alpha=0.50)
        label = f"ARIMA{model.order} (d=1)"
    else:
        # Fallback: ARIMA(0,1,0) = mean of first-differences as drift
        diff  = np.diff(ev_raw)
        drift = float(diff.mean())
        h     = np.arange(1, n_periods + 1)
        fc    = ev_raw[-1] + drift * h
        s     = float(np.std(diff))
        ci95  = np.c_[fc - 1.96  * s * np.sqrt(h), fc + 1.96  * s * np.sqrt(h)]
        ci50  = np.c_[fc - 0.674 * s * np.sqrt(h), fc + 0.674 * s * np.sqrt(h)]
        label = "ARIMA(0,1,0) fallback"

    return (
        np.maximum(fc, 0),
        np.maximum(ci95[:, 0], 0), ci95[:, 1],
        np.maximum(ci50[:, 0], 0), ci50[:, 1],
        label,
    )


# ── Prophet ───────────────────────────────────────────────────────────────────

def _fit_prophet(ev_raw, years_train, years_pred, exog_obs_raw, exog_fc_raw, use_exog):
    """
    Prophet on ORIGINAL scale.
    Small datasets explode on log scale; original scale gives stable results.
    Intervals are symmetric on the original scale.
    """
    df_train = pd.DataFrame({
        "ds": pd.to_datetime([f"{y}-07-01" for y in years_train]),
        "y":  ev_raw,
    })
    df_pred = pd.DataFrame({
        "ds": pd.to_datetime([f"{y}-07-01" for y in years_pred]),
    })

    m = Prophet(
        yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
        interval_width=0.95, uncertainty_samples=300,
        changepoint_prior_scale=0.3,   # moderate flexibility
    )

    if use_exog and exog_obs_raw is not None:
        for i, col in enumerate(EXOG_VARS):
            m.add_regressor(col, standardize=True)
            df_train[col] = exog_obs_raw[:, i]
            df_pred[col]  = exog_fc_raw[:len(years_pred), i]

    m.fit(df_train)
    fcast = m.predict(df_pred)

    fc   = np.maximum(fcast["yhat"].values,       0)
    lo95 = np.maximum(fcast["yhat_lower"].values, 0)
    hi95 = np.maximum(fcast["yhat_upper"].values, 0)

    # 50% CI by z-score scaling of 95% half-width
    hw95 = (hi95 - lo95) / 2
    hw50 = hw95 * (0.674 / 1.96)

    return (
        fc,
        lo95,                hi95,
        np.maximum(fc - hw50, 0), fc + hw50,
        "Prophet",
    )


# ── STS (Structural Time Series) ──────────────────────────────────────────────

def _fit_sts(ev_raw, exog_obs_s, exog_fc_s, use_exog, n_periods):
    """
    STS Smooth Trend on ORIGINAL scale via statsmodels UnobservedComponents.
    'smooth trend' = stochastic level + deterministic slope (stable with few points).
    conf_int() returns numpy array directly in statsmodels ≥ 0.14.
    """
    X_obs = exog_obs_s if use_exog else None
    X_fc  = exog_fc_s[:n_periods] if use_exog else None

    model = UnobservedComponents(ev_raw, level="smooth trend", exog=X_obs)
    try:
        res = model.fit(disp=False, maxiter=300)
    except Exception:
        res = model.fit(disp=False, method="nm", maxiter=500)

    fcast = res.get_forecast(steps=n_periods, exog=X_fc)
    fc    = np.maximum(np.asarray(fcast.predicted_mean), 0)
    ci95  = fcast.conf_int(alpha=0.05)   # numpy ndarray (n, 2), no .values
    ci50  = fcast.conf_int(alpha=0.50)

    return (
        fc,
        np.maximum(ci95[:, 0], 0), ci95[:, 1],
        np.maximum(ci50[:, 0], 0), ci50[:, 1],
        "STS (Smooth Trend)",
    )


# ── Dispatcher ────────────────────────────────────────────────────────────────

def _dispatch(model_choice, ev_raw, log_y, exog_obs_s, exog_fc_s, use_exog,
              n_periods, exog_obs_raw=None, exog_fc_raw=None,
              years_train=None, years_pred=None):
    """Route to the selected model backend."""
    if model_choice == "Prophet" and HAS_PROPHET:
        ytr = years_train or YEARS_OBS
        ypr = years_pred  or YEARS_FC[:n_periods]
        try:
            return _fit_prophet(ev_raw, ytr, ypr, exog_obs_raw, exog_fc_raw, use_exog)
        except Exception as e:
            st.warning(f"Prophet failed ({e}), falling back to Auto-ARIMA.")
    if model_choice == "STS (Structural)" and HAS_STS:
        try:
            return _fit_sts(ev_raw, exog_obs_s, exog_fc_s, use_exog, n_periods)
        except Exception as e:
            st.warning(f"STS failed ({e}), falling back to Auto-ARIMA.")
    # Default: Auto-ARIMA on original scale (d=1 for stability)
    return _fit_arima(ev_raw, exog_obs_s, exog_fc_s, use_exog, n_periods)


# ══════════════════════════════════════════════════════════════════════════════
# Cached model wrappers
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def cached_forecast(ev_t, exog_obs_t, exog_fc_t, use_exog, model_choice):
    ev          = np.maximum(np.array(ev_t, float), 1)
    exog_obs    = np.array(exog_obs_t)
    exog_fc     = np.array(exog_fc_t)
    exog_obs_s, mu, sig = z_normalize(exog_obs)
    exog_fc_s, _, _     = z_normalize(exog_fc, mu, sig)
    return _dispatch(
        model_choice, ev, np.log(ev),
        exog_obs_s, exog_fc_s, use_exog, N_FC,
        exog_obs_raw=exog_obs, exog_fc_raw=exog_fc,
    )


@st.cache_data(show_spinner=False)
def cached_baseline(ev_t: tuple) -> tuple:
    """Exact replica of ev_forecast_simulation.py (symmetric intervals)."""
    ev  = np.array(ev_t, float)
    h   = np.array(YEARS_FC) - 2024
    fc  = ev[-1] + BASELINE_TREND_STEP * h
    sig = BASELINE_C * h ** 1.5
    return (
        fc,
        np.maximum(0, fc - 1.96  * sig),
        fc + 1.96  * sig,
        np.maximum(0, fc - 0.674 * sig),
        fc + 0.674 * sig,
        f"Linear drift +{BASELINE_TREND_STEP}/yr  |  C={BASELINE_C}",
    )


@st.cache_data(show_spinner=False)
def cached_backtest(ev_t, exog_obs_t, split_idx, use_exog, model_choice):
    ev    = np.array(ev_t, float)
    exog  = np.array(exog_obs_t)
    n_te  = len(ev) - split_idx
    if n_te <= 0:
        return [], [], [], [], 0.0, 0.0, 0.0

    ev_tr   = np.maximum(ev[:split_idx], 1)
    exog_tr = exog[:split_idx]
    exog_te = exog[split_idx:]
    log_tr  = np.log(ev_tr)

    exog_tr_s, mu, sig = z_normalize(exog_tr)
    exog_te_s, _, _    = z_normalize(exog_te, mu, sig)

    ytr = YEARS_OBS[:split_idx]
    yte = YEARS_OBS[split_idx:]

    try:
        fc, lo95, hi95, *_ = _dispatch(
            model_choice, ev_tr, log_tr,
            exog_tr_s, exog_te_s, use_exog, n_te,
            exog_obs_raw=exog_tr, exog_fc_raw=exog_te,
            years_train=ytr, years_pred=yte,
        )
    except Exception:
        fc, lo95, hi95, *_ = _fit_arima(ev_tr, exog_tr_s, exog_te_s, use_exog, n_te)

    actual = ev[split_idx:]
    mae    = float(np.mean(np.abs(actual - fc)))
    rmse   = float(np.sqrt(np.mean((actual - fc) ** 2)))
    mape   = float(np.mean(np.abs((actual - fc) / np.maximum(actual, 1))) * 100)
    return list(actual), list(fc), list(lo95), list(hi95), mae, mape, rmse


# ══════════════════════════════════════════════════════════════════════════════
# Page layout
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="⚡ EV Forecast", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("📊 Observed EV Stock (units)")
    ev_vals = [
        st.number_input(str(yr), min_value=0, max_value=500000,
                        value=EV_DEFAULT[i], step=100, key=f"ev_{yr}")
        for i, yr in enumerate(YEARS_OBS)
    ]

    st.subheader("🔧 Model Settings")
    model_choice = st.selectbox(
        "Forecast Model",
        AVAILABLE_MODELS,
        index=0,
        help="Auto-ARIMA: AIC-optimal ARIMA with optional exog regressors\n"
             "Prophet: Facebook Prophet with annual changepoints\n"
             "STS: Local Linear Trend via statsmodels",
    )
    use_exog = st.checkbox("Use exogenous variables", value=True,
                           disabled=(model_choice == "Fallback (log-linear)"))
    split_yr = st.select_slider("Backtest: training end year",
                                options=YEARS_OBS[1:-1], value=2022)

    st.subheader("🌐 Main Scenario")
    main_sc = st.selectbox("Scenario", list(SCENARIOS.keys()), index=0)

    with st.expander("Customize scenario parameters (2030 value)"):
        sc_params: dict[str, list] = {}
        for var, (lo_b, hi_b, step_b) in EXOG_BOUNDS.items():
            default_end = float(SCENARIOS[main_sc][var][-1])
            end_val     = st.slider(var, lo_b, hi_b, default_end, step_b)
            start_val   = float(SCENARIOS[main_sc][var][0])
            sc_params[var] = [
                round(start_val + (end_val - start_val) * i / (N_FC - 1), 2)
                for i in range(N_FC)
            ]

# ── Compute ───────────────────────────────────────────────────────────────────
exog_obs_arr = build_exog(EXOG_OBS)
exog_fc_arr  = build_exog(sc_params)

_spinner_msg = {
    "Auto-ARIMA":       "Training Auto-ARIMA...",
    "Prophet":          "Fitting Prophet...",
    "STS (Structural)": "Fitting STS (Local Linear Trend)...",
}.get(model_choice, "Computing forecast...")

# Main scenario forecast
if main_sc == "Baseline":
    fc, lo95, hi95, lo50, hi50, order_str = cached_baseline(tuple(ev_vals))
else:
    with st.spinner(_spinner_msg):
        fc, lo95, hi95, lo50, hi50, order_str = cached_forecast(
            tuple(ev_vals), to_t(exog_obs_arr), to_t(exog_fc_arr),
            use_exog, model_choice,
        )

# Backtest
split_idx = YEARS_OBS.index(split_yr) + 1
with st.spinner(f"Backtesting ({model_choice})..."):
    bt_actual, bt_fc, bt_lo, bt_hi, bt_mae, bt_mape, bt_rmse = cached_backtest(
        tuple(ev_vals), to_t(exog_obs_arr), split_idx, use_exog,
        model_choice if main_sc != "Baseline" else "Auto-ARIMA",
    )

# All-scenario results (using selected model for non-Baseline)
sc_results: dict[str, tuple] = {}
for sc_name, sc_exog in SCENARIOS.items():
    if sc_name == "Baseline":
        sc_results[sc_name] = cached_baseline(tuple(ev_vals))
    else:
        exog_fc_sc = build_exog(sc_exog)
        sc_results[sc_name] = cached_forecast(
            tuple(ev_vals), to_t(exog_obs_arr), to_t(exog_fc_sc),
            use_exog, model_choice,
        )

# Auto-ARIMA Baseline overlay (always computed for comparison)
arima_bl_fc, arima_bl_lo95, arima_bl_hi95, arima_bl_lo50, arima_bl_hi50, _ = (
    cached_forecast(
        tuple(ev_vals), to_t(exog_obs_arr),
        to_t(build_exog(SCENARIOS["Baseline"])),
        use_exog, "Auto-ARIMA",
    )
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚡ EV Stock Forecast Simulation")

if main_sc == "Baseline":
    caps = [
        "Model: **Reference (matches ev_forecast_simulation.py)**",
        f"Spec: **{order_str}**",
        "Intervals: **Symmetric ± z·C·h^1.5**",
        f"Scenario: **{main_sc}**",
    ]
else:
    caps = [
        f"Model: **{model_choice}**",
        f"Spec: **{order_str}**",
        f"Exogenous Vars: **{'Enabled' if use_exog else 'Disabled'}**",
        f"Scenario: **{main_sc}**",
    ]
st.caption("　|　".join(caps))

missing = [m for m in _ALL_MODELS if m not in AVAILABLE_MODELS]
if missing:
    st.info(f"Install to unlock: `pip install {' '.join(m.split()[0].lower() for m in missing)}`")

# ── Interval band label (by model) ───────────────────────────────────────────
_band_label = {
    "Auto-ARIMA":       "95% Prediction Interval (ARIMA, symmetric)",
    "Prophet":          "95% Prediction Interval (Prophet, symmetric)",
    "STS (Structural)": "95% Prediction Interval (STS, symmetric)",
}.get(model_choice, "95% Prediction Interval")
if main_sc == "Baseline":
    _band_label = "95% Prediction Interval (Reference, symmetric)"

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Forecast", "🔍 Backtest", "🌐 Scenario Comparison", "💾 Data & Export"]
)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: Forecast
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    fig = go.Figure()

    # 95% band
    fig.add_trace(go.Scatter(
        x=YEARS_FC + YEARS_FC[::-1],
        y=list(hi95) + list(lo95[::-1]),
        fill="toself", fillcolor="rgba(135,206,235,0.30)",
        line=dict(color="rgba(0,0,0,0)"),
        name=_band_label, hoverinfo="skip",
    ))
    # 50% band
    fig.add_trace(go.Scatter(
        x=YEARS_FC + YEARS_FC[::-1],
        y=list(hi50) + list(lo50[::-1]),
        fill="toself", fillcolor="rgba(30,144,255,0.35)",
        line=dict(color="rgba(0,0,0,0)"),
        name="50% Prediction Interval", hoverinfo="skip",
    ))
    # Observed
    fig.add_trace(go.Scatter(
        x=YEARS_OBS, y=ev_vals,
        mode="lines+markers",
        line=dict(color="black", width=2),
        marker=dict(size=9, color="black"),
        name="Observed EV Stock",
    ))
    # Bridge 2024→2025
    fig.add_trace(go.Scatter(
        x=[YEARS_OBS[-1], YEARS_FC[0]], y=[ev_vals[-1], float(fc[0])],
        mode="lines", line=dict(color="black", width=2),
        showlegend=False, hoverinfo="skip",
    ))
    # Main forecast
    fig.add_trace(go.Scatter(
        x=YEARS_FC, y=list(fc),
        mode="lines+markers",
        line=dict(color="deepskyblue", width=2, dash="dash"),
        marker=dict(size=11, color="deepskyblue"),
        name=f"{model_choice if main_sc != 'Baseline' else 'Reference'} ({main_sc})",
        customdata=[[round(l), round(h)] for l, h in zip(lo95, hi95)],
        hovertemplate="%{x}: %{y:,.0f} units<br>95% CI: [%{customdata[0]:,}, %{customdata[1]:,}]<extra></extra>",
    ))

    # ARIMA Baseline overlay (only when Baseline scenario is selected)
    if main_sc == "Baseline":
        fig.add_trace(go.Scatter(
            x=[YEARS_OBS[-1], YEARS_FC[0]],
            y=[ev_vals[-1], float(arima_bl_fc[0])],
            mode="lines", line=dict(color="royalblue", width=1.5, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=YEARS_FC, y=list(arima_bl_fc),
            mode="lines+markers",
            line=dict(color="royalblue", width=1.5, dash="dot"),
            marker=dict(size=8, color="royalblue", symbol="diamond"),
            name="Auto-ARIMA Baseline (comparison)",
            customdata=[[round(l), round(h)] for l, h in zip(arima_bl_lo95, arima_bl_hi95)],
            hovertemplate="%{x}: %{y:,.0f} units (ARIMA)<br>95% CI: [%{customdata[0]:,}, %{customdata[1]:,}]<extra></extra>",
        ))

    fig.add_vline(x=2024.5, line_dash="dash", line_color="gray", line_width=1.5)
    fig.add_annotation(x=2021.5, y=1.07, xref="x", yref="paper",
                       text="Observed Window", showarrow=False, font=dict(size=13))
    fig.add_annotation(x=2027.5, y=1.07, xref="x", yref="paper",
                       text="Forecast Window", showarrow=False, font=dict(size=13))
    fig.update_layout(
        title=f"EV Stock Forecast — {main_sc} Scenario  [{model_choice if main_sc != 'Baseline' else 'Reference'}]",
        xaxis_title="Year", yaxis_title="Total Registered EV Stock (units)",
        hovermode="x unified", height=540,
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Table")
    fc_tbl = pd.DataFrame({
        "Year":            YEARS_FC,
        "Forecast (Med.)": [f"{v:,.0f}" for v in fc],
        "95% Lower":       [f"{v:,.0f}" for v in lo95],
        "95% Upper":       [f"{v:,.0f}" for v in hi95],
        "50% Lower":       [f"{v:,.0f}" for v in lo50],
        "50% Upper":       [f"{v:,.0f}" for v in hi50],
    })
    if main_sc == "Baseline":
        fc_tbl["Auto-ARIMA Baseline"] = [f"{v:,.0f}" for v in arima_bl_fc]
    st.dataframe(fc_tbl.set_index("Year"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: Backtest
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    train_years = YEARS_OBS[:split_idx]
    test_years  = YEARS_OBS[split_idx:]

    _bt_model = model_choice if main_sc != "Baseline" else "Auto-ARIMA"
    st.subheader(f"Backtest [{_bt_model}]  —  Train: 2019–{split_yr}  /  Test: {split_yr + 1}–2024")

    if test_years:
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE",  f"{bt_mae:,.0f} units")
        c2.metric("MAPE", f"{bt_mape:.1f} %")
        c3.metric("RMSE", f"{bt_rmse:,.0f} units")
    else:
        st.info("No test period. Select an earlier split year.")

    fig_bt = go.Figure()
    if test_years:
        fig_bt.add_trace(go.Scatter(
            x=test_years + test_years[::-1],
            y=bt_hi + bt_lo[::-1],
            fill="toself", fillcolor="rgba(255,165,0,0.25)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Prediction Interval (Backtest)", hoverinfo="skip",
        ))
    fig_bt.add_trace(go.Scatter(
        x=train_years, y=ev_vals[:split_idx],
        mode="lines+markers", line=dict(color="black", width=2),
        marker=dict(size=9), name="Training Data",
    ))
    if test_years:
        fig_bt.add_trace(go.Scatter(
            x=test_years, y=bt_actual,
            mode="lines+markers",
            line=dict(color="black", width=2, dash="dot"),
            marker=dict(size=9, symbol="diamond"),
            name="Test Actual",
        ))
        fig_bt.add_trace(go.Scatter(
            x=test_years, y=bt_fc,
            mode="lines+markers",
            line=dict(color="orange", width=2, dash="dash"),
            marker=dict(size=10),
            name=f"{_bt_model} Forecast",
        ))
    fig_bt.add_vline(x=split_yr + 0.5, line_dash="dash", line_color="gray")
    fig_bt.update_layout(
        title=f"Backtest: {_bt_model} Forecast vs. Actual",
        xaxis_title="Year", yaxis_title="Total Registered EV Stock (units)",
        hovermode="x unified", height=460,
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    if test_years:
        bt_tbl = pd.DataFrame({
            "Year":           test_years,
            "Actual":         [f"{v:,.0f}" for v in bt_actual],
            "Forecast":       [f"{v:,.0f}" for v in bt_fc],
            "Error (units)":  [f"{a - p:+,.0f}" for a, p in zip(bt_actual, bt_fc)],
            "Error Rate (%)": [f"{(a-p)/max(a,1)*100:+.1f}" for a, p in zip(bt_actual, bt_fc)],
        })
        st.dataframe(bt_tbl.set_index("Year"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3: Scenario comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"All Scenario Comparison  [{model_choice}]")

    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=YEARS_OBS, y=ev_vals,
        mode="lines+markers", line=dict(color="black", width=2),
        marker=dict(size=9), name="Observed Data",
    ))
    for sc_name, res in sc_results.items():
        sc_fc = res[0]
        color = SC_COLORS.get(sc_name, "gray")
        fig_sc.add_trace(go.Scatter(
            x=[YEARS_OBS[-1], YEARS_FC[0]], y=[ev_vals[-1], float(sc_fc[0])],
            mode="lines", line=dict(color=color, width=1.5),
            showlegend=False, hoverinfo="skip",
        ))
        fig_sc.add_trace(go.Scatter(
            x=YEARS_FC, y=list(sc_fc),
            mode="lines+markers",
            line=dict(color=color, width=2,
                      dash="solid" if sc_name == "Baseline" else "dash"),
            marker=dict(size=8),
            name=sc_name,
        ))

    # Auto-ARIMA Baseline overlay
    fig_sc.add_trace(go.Scatter(
        x=[YEARS_OBS[-1], YEARS_FC[0]], y=[ev_vals[-1], float(arima_bl_fc[0])],
        mode="lines", line=dict(color="royalblue", width=1.5, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))
    fig_sc.add_trace(go.Scatter(
        x=YEARS_FC, y=list(arima_bl_fc),
        mode="lines+markers",
        line=dict(color="royalblue", width=2, dash="dot"),
        marker=dict(size=8, symbol="diamond", color="royalblue"),
        name="Auto-ARIMA Baseline",
    ))

    fig_sc.add_vline(x=2024.5, line_dash="dash", line_color="gray")
    fig_sc.update_layout(
        title=f"EV Stock Forecast by Scenario (2025–2030)  [{model_choice}]",
        xaxis_title="Year", yaxis_title="Total Registered EV Stock (units)",
        hovermode="x unified", height=510,
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    sc_tbl = pd.DataFrame({"Year": YEARS_FC})
    for sc_name, res in sc_results.items():
        sc_tbl[sc_name] = [f"{v:,.0f}" for v in res[0]]
    sc_tbl["Auto-ARIMA Baseline"] = [f"{v:,.0f}" for v in arima_bl_fc]
    st.dataframe(sc_tbl.set_index("Year"), use_container_width=True)

    st.subheader("Exogenous Variable Assumptions by Scenario")
    cols = st.columns(2)
    for i, var in enumerate(EXOG_VARS):
        with cols[i % 2]:
            fig_ev = go.Figure()
            fig_ev.add_trace(go.Scatter(
                x=YEARS_OBS, y=EXOG_OBS[var],
                mode="lines+markers", line=dict(color="black", width=2),
                marker=dict(size=7), name="Observed",
            ))
            for sc_name, sc_exog in SCENARIOS.items():
                fig_ev.add_trace(go.Scatter(
                    x=YEARS_FC, y=sc_exog[var],
                    mode="lines+markers",
                    line=dict(color=SC_COLORS[sc_name], width=1.5, dash="dash"),
                    marker=dict(size=6), name=sc_name,
                    showlegend=(i == 0),
                ))
            fig_ev.add_vline(x=2024.5, line_dash="dash", line_color="gray")
            fig_ev.update_layout(title=var, height=270,
                                 margin=dict(t=36, b=10, l=10, r=10))
            st.plotly_chart(fig_ev, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 4: Data & Export
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Observed Data")
    obs_df = pd.DataFrame({"Year": YEARS_OBS, "EV Stock (units)": ev_vals})
    for var in EXOG_VARS:
        obs_df[var] = EXOG_OBS[var]
    st.dataframe(obs_df.set_index("Year"), use_container_width=True)

    st.subheader(f"Forecast Data — {main_sc} Scenario [{model_choice if main_sc != 'Baseline' else 'Reference'}]")
    fc_export = pd.DataFrame({
        "Year":      YEARS_FC,
        "Scenario":  main_sc,
        "Model":     model_choice if main_sc != "Baseline" else "Reference",
        "Forecast":  [round(v) for v in fc],
        "95% Lower": [round(v) for v in lo95],
        "95% Upper": [round(v) for v in hi95],
        "50% Lower": [round(v) for v in lo50],
        "50% Upper": [round(v) for v in hi50],
    })
    for var in EXOG_VARS:
        fc_export[var] = sc_params[var]
    st.dataframe(fc_export.set_index("Year"), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "📥 Observed Data CSV",
            obs_df.to_csv(index=False).encode("utf-8-sig"),
            "ev_observed.csv", "text/csv",
        )
    with c2:
        st.download_button(
            f"📥 Forecast CSV ({main_sc})",
            fc_export.to_csv(index=False).encode("utf-8-sig"),
            "ev_forecast.csv", "text/csv",
        )
    with c3:
        all_sc_df = pd.DataFrame({"Year": YEARS_FC})
        for sc_name, res in sc_results.items():
            all_sc_df[f"{sc_name} Forecast"]   = [round(v) for v in res[0]]
            all_sc_df[f"{sc_name} 95% Lower"]  = [round(v) for v in res[1]]
            all_sc_df[f"{sc_name} 95% Upper"]  = [round(v) for v in res[2]]
        all_sc_df["Auto-ARIMA Baseline Forecast"]  = [round(v) for v in arima_bl_fc]
        all_sc_df["Auto-ARIMA Baseline 95% Lower"] = [round(v) for v in arima_bl_lo95]
        all_sc_df["Auto-ARIMA Baseline 95% Upper"] = [round(v) for v in arima_bl_hi95]
        st.download_button(
            "📥 All Scenarios CSV",
            all_sc_df.to_csv(index=False).encode("utf-8-sig"),
            "ev_all_scenarios.csv", "text/csv",
        )
