"""
EV Stock Forecast Streamlit App
- Auto-ARIMA (pmdarima) with exogenous variables
- Log-normal prediction intervals
- Backtest evaluation (MAE / MAPE / RMSE)
- 4-scenario analysis
- Interactive Plotly charts
- CSV export
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    from pmdarima import auto_arima
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════
YEARS_OBS = [2019, 2020, 2021, 2022, 2023, 2024]
YEARS_FC  = [2025, 2026, 2027, 2028, 2029, 2030]
N_FC      = len(YEARS_FC)

EV_DEFAULT = [100, 500, 1500, 3500, 5000, 5000]

EXOG_VARS = [
    "補助金額（万円/台）",
    "充電インフラ数（基）",
    "ガソリン価格（円/L）",
    "EV車種数",
    "GDP成長率（%）",
]

# Observed values of external variables (2019-2024)
EXOG_OBS: dict[str, list] = {
    "補助金額（万円/台）":  [0,    10,   50,    100,   80,    60   ],
    "充電インフラ数（基）": [1000, 2000, 5000,  10000, 15000, 20000],
    "ガソリン価格（円/L）": [150,  130,  145,   170,   175,   165  ],
    "EV車種数":            [3,    5,    8,     15,    20,    25   ],
    "GDP成長率（%）":       [0.5, -4.1,  2.1,   1.0,   1.9,   0.5  ],
}

# Scenario assumptions for external variables (2025-2030)
SCENARIOS: dict[str, dict[str, list]] = {
    "ベースライン": {
        "補助金額（万円/台）":  [55,    50,    45,    40,    35,    30   ],
        "充電インフラ数（基）": [25000, 30000, 35000, 40000, 45000, 50000],
        "ガソリン価格（円/L）": [165,   167,   168,   170,   170,   170  ],
        "EV車種数":            [28,    32,    36,    40,    44,    48   ],
        "GDP成長率（%）":       [1.0,   1.0,   1.0,   1.0,   1.0,   1.0  ],
    },
    "政策強化": {
        "補助金額（万円/台）":  [70,    75,    80,    80,    80,    80   ],
        "充電インフラ数（基）": [30000, 40000, 52000, 62000, 72000, 80000],
        "ガソリン価格（円/L）": [175,   180,   185,   190,   195,   200  ],
        "EV車種数":            [32,    38,    44,    50,    56,    60   ],
        "GDP成長率（%）":       [1.5,   1.5,   1.5,   1.5,   1.5,   1.5  ],
    },
    "インフラ制約": {
        "補助金額（万円/台）":  [50,    45,    35,    25,    20,    20   ],
        "充電インフラ数（基）": [21000, 22500, 24000, 25500, 27000, 28000],
        "ガソリン価格（円/L）": [160,   158,   157,   155,   155,   155  ],
        "EV車種数":            [26,    28,    31,    34,    37,    40   ],
        "GDP成長率（%）":       [0.5,   0.5,   0.5,   0.5,   0.5,   0.5  ],
    },
    "価格急落": {
        "補助金額（万円/台）":  [50,    40,    30,    20,    20,    20   ],
        "充電インフラ数（基）": [26000, 32000, 38000, 46000, 52000, 58000],
        "ガソリン価格（円/L）": [180,   190,   200,   210,   215,   220  ],
        "EV車種数":            [35,    44,    52,    60,    65,    70   ],
        "GDP成長率（%）":       [1.2,   1.2,   1.2,   1.2,   1.2,   1.2  ],
    },
}

SC_COLORS = {
    "ベースライン":  "royalblue",
    "政策強化":      "green",
    "インフラ制約":  "orange",
    "価格急落":      "purple",
}

EXOG_BOUNDS = {
    "補助金額（万円/台）":  (0.0,   120.0,  1.0),
    "充電インフラ数（基）": (1000., 150000., 1000.),
    "ガソリン価格（円/L）": (80.0,  400.0,   5.0),
    "EV車種数":            (1.0,   150.0,   1.0),
    "GDP成長率（%）":       (-5.0,  15.0,    0.1),
}

# ══════════════════════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════════════════════

def z_normalize(
    arr: np.ndarray,
    mu: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(arr, float)
    if mu is None:
        mu    = arr.mean(axis=0)
        sigma = arr.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (arr - mu) / sigma, mu, sigma


def build_exog(var_dict: dict[str, list]) -> np.ndarray:
    return np.column_stack([var_dict[k] for k in EXOG_VARS])


def fit_and_predict(
    log_y: np.ndarray,
    exog_obs_s: np.ndarray | None,
    exog_fc_s: np.ndarray | None,
    use_exog: bool,
    n_periods: int = N_FC,
) -> tuple:
    """
    Fit ARIMA on log scale, return forecasts transformed back via exp().
    Prediction intervals are inherently log-normal (asymmetric).
    Returns: (fc, lo95, hi95, lo50, hi50, order_str)
    """
    X_obs = exog_obs_s if use_exog else None
    X_fc  = exog_fc_s  if use_exog else None

    if HAS_PMDARIMA:
        model = auto_arima(
            log_y, X=X_obs,
            seasonal=False, stepwise=True,
            information_criterion="aic",
            max_p=2, max_q=2, max_d=1,
            error_action="ignore", suppress_warnings=True,
        )
        fc_log, ci95 = model.predict(
            n_periods, X=X_fc, return_conf_int=True, alpha=0.05
        )
        _,      ci50 = model.predict(
            n_periods, X=X_fc, return_conf_int=True, alpha=0.50
        )
        order_str = str(model.order)
    else:
        # Fallback: log-linear trend with expanding intervals
        x = np.arange(len(log_y))
        p = np.polyfit(x, log_y, 1)
        x_fc   = np.arange(len(log_y), len(log_y) + n_periods)
        fc_log = np.polyval(p, x_fc)
        s      = np.std(log_y - np.polyval(p, x))
        h      = np.arange(1, n_periods + 1)
        ci95   = np.column_stack([fc_log - 1.96  * s * np.sqrt(h),
                                  fc_log + 1.96  * s * np.sqrt(h)])
        ci50   = np.column_stack([fc_log - 0.674 * s * np.sqrt(h),
                                  fc_log + 0.674 * s * np.sqrt(h)])
        order_str = "fallback"

    return (
        np.exp(fc_log),
        np.exp(ci95[:, 0]), np.exp(ci95[:, 1]),
        np.exp(ci50[:, 0]), np.exp(ci50[:, 1]),
        order_str,
    )


# ── Cached wrappers ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_forecast(
    ev_t: tuple, exog_obs_t: tuple, exog_fc_t: tuple, use_exog: bool
) -> tuple:
    ev       = np.maximum(np.array(ev_t, float), 1)
    exog_obs = np.array(exog_obs_t)
    exog_fc  = np.array(exog_fc_t)
    exog_obs_s, mu, sig = z_normalize(exog_obs)
    exog_fc_s, _, _     = z_normalize(exog_fc, mu, sig)
    return fit_and_predict(np.log(ev), exog_obs_s, exog_fc_s, use_exog)


@st.cache_data(show_spinner=False)
def cached_backtest(
    ev_t: tuple, exog_obs_t: tuple, split_idx: int, use_exog: bool
) -> tuple:
    ev      = np.array(ev_t, float)
    exog    = np.array(exog_obs_t)
    n_te    = len(ev) - split_idx

    if n_te <= 0:
        return [], [], [], [], 0.0, 0.0, 0.0

    ev_tr   = np.maximum(ev[:split_idx], 1)
    exog_tr = exog[:split_idx]
    exog_te = exog[split_idx:]

    exog_tr_s, mu, sig = z_normalize(exog_tr)
    exog_te_s, _, _    = z_normalize(exog_te, mu, sig)

    fc, lo95, hi95, *_ = fit_and_predict(
        np.log(ev_tr), exog_tr_s, exog_te_s, use_exog, n_periods=n_te
    )

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
    st.header("⚙️ 設定")

    # Observed EV stock
    st.subheader("📊 観測 EV 台数（台）")
    ev_vals = [
        st.number_input(
            f"{yr}年", min_value=0, max_value=500000,
            value=EV_DEFAULT[i], step=100, key=f"ev_{yr}"
        )
        for i, yr in enumerate(YEARS_OBS)
    ]

    # Model settings
    st.subheader("🔧 モデル設定")
    use_exog = st.checkbox("外生変数を使用する（ARIMAX）", value=True)
    split_yr = st.select_slider(
        "バックテスト：学習終了年",
        options=YEARS_OBS[1:-1], value=2022,
    )

    # Scenario selector
    st.subheader("🌐 メインシナリオ")
    main_sc = st.selectbox("シナリオ", list(SCENARIOS.keys()), index=0)

    # Customise scenario end-values
    with st.expander("シナリオパラメータのカスタマイズ（2030年値）"):
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

def to_t(arr: np.ndarray) -> tuple:
    return tuple(map(tuple, arr.tolist()))

with st.spinner("Auto-ARIMA モデルを学習・予測中..."):
    fc, lo95, hi95, lo50, hi50, order_str = cached_forecast(
        tuple(ev_vals), to_t(exog_obs_arr), to_t(exog_fc_arr), use_exog
    )

split_idx = YEARS_OBS.index(split_yr) + 1
with st.spinner("バックテストを実行中..."):
    bt_actual, bt_fc, bt_lo, bt_hi, bt_mae, bt_mape, bt_rmse = cached_backtest(
        tuple(ev_vals), to_t(exog_obs_arr), split_idx, use_exog
    )

sc_results: dict[str, tuple] = {}
for sc_name, sc_exog in SCENARIOS.items():
    exog_fc_sc = build_exog(sc_exog)
    res = cached_forecast(
        tuple(ev_vals), to_t(exog_obs_arr), to_t(exog_fc_sc), use_exog
    )
    sc_results[sc_name] = res

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚡ EV 登録台数予測シミュレーション")

caps = [
    f"モデル: **{'Auto-ARIMA' if HAS_PMDARIMA else 'Fallback (log-linear)'}**",
    f"ARIMA次数: **{order_str}**",
    f"外生変数: **{'使用' if use_exog else '未使用'}**",
    f"シナリオ: **{main_sc}**",
]
st.caption("　|　".join(caps))

if not HAS_PMDARIMA:
    st.warning(
        "pmdarima が見つかりません。`pip install pmdarima` を実行するとAuto-ARIMAが有効になります。"
    )

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 予測チャート", "🔍 バックテスト", "🌐 シナリオ比較", "💾 データ・エクスポート"]
)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: Forecast
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    fig = go.Figure()

    # 95% prediction band (log-normal → asymmetric)
    fig.add_trace(go.Scatter(
        x=YEARS_FC + YEARS_FC[::-1],
        y=list(hi95) + list(lo95[::-1]),
        fill="toself", fillcolor="rgba(135,206,235,0.30)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% 予測区間（対数正規）", hoverinfo="skip",
    ))
    # 50% prediction band
    fig.add_trace(go.Scatter(
        x=YEARS_FC + YEARS_FC[::-1],
        y=list(hi50) + list(lo50[::-1]),
        fill="toself", fillcolor="rgba(30,144,255,0.35)",
        line=dict(color="rgba(0,0,0,0)"),
        name="50% 予測区間", hoverinfo="skip",
    ))
    # Observed
    fig.add_trace(go.Scatter(
        x=YEARS_OBS, y=ev_vals,
        mode="lines+markers",
        line=dict(color="black", width=2),
        marker=dict(size=9, color="black"),
        name="観測 EV 台数",
    ))
    # Bridge 2024 → 2025
    fig.add_trace(go.Scatter(
        x=[YEARS_OBS[-1], YEARS_FC[0]],
        y=[ev_vals[-1], float(fc[0])],
        mode="lines", line=dict(color="black", width=2),
        showlegend=False, hoverinfo="skip",
    ))
    # ARIMA forecast
    fig.add_trace(go.Scatter(
        x=YEARS_FC, y=list(fc),
        mode="lines+markers",
        line=dict(color="royalblue", width=2, dash="dash"),
        marker=dict(size=11, color="blue"),
        name=f"Auto-ARIMA（{main_sc}）",
        customdata=[[round(l), round(h)] for l, h in zip(lo95, hi95)],
        hovertemplate=(
            "%{x}年: %{y:,.0f} 台<br>"
            "95%CI: [%{customdata[0]:,}, %{customdata[1]:,}]<extra></extra>"
        ),
    ))

    fig.add_vline(x=2024.5, line_dash="dash", line_color="gray", line_width=1.5)
    fig.add_annotation(x=2021.5, y=1.07, xref="x", yref="paper",
                       text="観測ウィンドウ", showarrow=False,
                       font=dict(size=13, color="black"))
    fig.add_annotation(x=2027.5, y=1.07, xref="x", yref="paper",
                       text="予測ウィンドウ", showarrow=False,
                       font=dict(size=13, color="black"))
    fig.update_layout(
        title=f"EV 登録台数予測（{main_sc}シナリオ）",
        xaxis_title="年", yaxis_title="EV 登録台数（台）",
        hovermode="x unified", height=530,
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("予測値テーブル")
    fc_tbl = pd.DataFrame({
        "年":          YEARS_FC,
        "予測（中央値）": [f"{v:,.0f}" for v in fc],
        "95% 下限":    [f"{v:,.0f}" for v in lo95],
        "95% 上限":    [f"{v:,.0f}" for v in hi95],
        "50% 下限":    [f"{v:,.0f}" for v in lo50],
        "50% 上限":    [f"{v:,.0f}" for v in hi50],
    })
    st.dataframe(fc_tbl.set_index("年"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: Backtest
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    train_years = YEARS_OBS[:split_idx]
    test_years  = YEARS_OBS[split_idx:]

    st.subheader(
        f"バックテスト（学習: 2019〜{split_yr}年 ／ テスト: {split_yr+1}〜2024年）"
    )

    if test_years:
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE",  f"{bt_mae:,.0f} 台")
        c2.metric("MAPE", f"{bt_mape:.1f} %")
        c3.metric("RMSE", f"{bt_rmse:,.0f} 台")
    else:
        st.info("テスト期間がありません。分割年を早めてください。")

    fig_bt = go.Figure()
    if test_years:
        fig_bt.add_trace(go.Scatter(
            x=test_years + test_years[::-1],
            y=bt_hi + bt_lo[::-1],
            fill="toself", fillcolor="rgba(255,165,0,0.25)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% 予測区間（バックテスト）", hoverinfo="skip",
        ))
    fig_bt.add_trace(go.Scatter(
        x=train_years, y=ev_vals[:split_idx],
        mode="lines+markers", line=dict(color="black", width=2),
        marker=dict(size=9), name="学習データ",
    ))
    if test_years:
        fig_bt.add_trace(go.Scatter(
            x=test_years, y=bt_actual,
            mode="lines+markers",
            line=dict(color="black", width=2, dash="dot"),
            marker=dict(size=9, symbol="diamond"),
            name="テスト実績",
        ))
        fig_bt.add_trace(go.Scatter(
            x=test_years, y=bt_fc,
            mode="lines+markers",
            line=dict(color="orange", width=2, dash="dash"),
            marker=dict(size=10),
            name="モデル予測",
        ))

    fig_bt.add_vline(x=split_yr + 0.5, line_dash="dash", line_color="gray")
    fig_bt.update_layout(
        title="バックテスト：モデル予測 vs 実績",
        xaxis_title="年", yaxis_title="EV 登録台数（台）",
        hovermode="x unified", height=460,
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    if test_years:
        bt_tbl = pd.DataFrame({
            "年":       test_years,
            "実績":     [f"{v:,.0f}" for v in bt_actual],
            "予測":     [f"{v:,.0f}" for v in bt_fc],
            "誤差（台）": [f"{a-p:+,.0f}" for a, p in zip(bt_actual, bt_fc)],
            "誤差率（%）": [
                f"{(a-p)/max(a,1)*100:+.1f}"
                for a, p in zip(bt_actual, bt_fc)
            ],
        })
        st.dataframe(bt_tbl.set_index("年"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3: Scenario comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("全シナリオ比較")

    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=YEARS_OBS, y=ev_vals,
        mode="lines+markers", line=dict(color="black", width=2),
        marker=dict(size=9), name="観測データ",
    ))
    for sc_name, res in sc_results.items():
        sc_fc = res[0]
        color = SC_COLORS.get(sc_name, "gray")
        fig_sc.add_trace(go.Scatter(
            x=[YEARS_OBS[-1], YEARS_FC[0]],
            y=[ev_vals[-1], float(sc_fc[0])],
            mode="lines", line=dict(color=color, width=1.5),
            showlegend=False, hoverinfo="skip",
        ))
        fig_sc.add_trace(go.Scatter(
            x=YEARS_FC, y=list(sc_fc),
            mode="lines+markers",
            line=dict(
                color=color, width=2,
                dash="solid" if sc_name == "ベースライン" else "dash",
            ),
            marker=dict(size=8),
            name=sc_name,
        ))

    fig_sc.add_vline(x=2024.5, line_dash="dash", line_color="gray")
    fig_sc.update_layout(
        title="シナリオ別 EV 登録台数予測（2025〜2030年）",
        xaxis_title="年", yaxis_title="EV 登録台数（台）",
        hovermode="x unified", height=510,
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # Scenario forecast table
    sc_tbl = pd.DataFrame({"年": YEARS_FC})
    for sc_name, res in sc_results.items():
        sc_tbl[sc_name] = [f"{v:,.0f}" for v in res[0]]
    st.dataframe(sc_tbl.set_index("年"), use_container_width=True)

    # External variable assumptions
    st.subheader("外生変数の想定値（シナリオ別）")
    cols = st.columns(2)
    for i, var in enumerate(EXOG_VARS):
        with cols[i % 2]:
            fig_ev = go.Figure()
            fig_ev.add_trace(go.Scatter(
                x=YEARS_OBS, y=EXOG_OBS[var],
                mode="lines+markers", line=dict(color="black", width=2),
                marker=dict(size=7), name="観測",
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
            fig_ev.update_layout(
                title=var, height=270,
                margin=dict(t=36, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_ev, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 4: Data & Export
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("観測データ")
    obs_df = pd.DataFrame({"年": YEARS_OBS, "EV登録台数": ev_vals})
    for var in EXOG_VARS:
        obs_df[var] = EXOG_OBS[var]
    st.dataframe(obs_df.set_index("年"), use_container_width=True)

    st.subheader(f"予測データ（{main_sc}シナリオ）")
    fc_export = pd.DataFrame({
        "年":       YEARS_FC,
        "シナリオ": main_sc,
        "予測台数": [round(v) for v in fc],
        "95%下限":  [round(v) for v in lo95],
        "95%上限":  [round(v) for v in hi95],
        "50%下限":  [round(v) for v in lo50],
        "50%上限":  [round(v) for v in hi50],
    })
    for var in EXOG_VARS:
        fc_export[var] = sc_params[var]
    st.dataframe(fc_export.set_index("年"), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "📥 観測データ CSV",
            obs_df.to_csv(index=False).encode("utf-8-sig"),
            "ev_observed.csv", "text/csv",
        )
    with c2:
        st.download_button(
            f"📥 予測 CSV（{main_sc}）",
            fc_export.to_csv(index=False).encode("utf-8-sig"),
            "ev_forecast.csv", "text/csv",
        )
    with c3:
        all_sc_df = pd.DataFrame({"年": YEARS_FC})
        for sc_name, res in sc_results.items():
            sc_fc_arr, sc_lo, sc_hi = res[0], res[1], res[2]
            all_sc_df[f"{sc_name}_予測"]   = [round(v) for v in sc_fc_arr]
            all_sc_df[f"{sc_name}_95%下限"] = [round(v) for v in sc_lo]
            all_sc_df[f"{sc_name}_95%上限"] = [round(v) for v in sc_hi]
        st.download_button(
            "📥 全シナリオ CSV",
            all_sc_df.to_csv(index=False).encode("utf-8-sig"),
            "ev_all_scenarios.csv", "text/csv",
        )
