# EV Stock Forecast App — User Guide

A Streamlit-based interactive dashboard for forecasting electric vehicle (EV) stock.  
Choose from four forecast models, compare four policy scenarios, run a backtest, and export results to CSV.

---

## Requirements

| Package | Version | Role |
|---------|---------|------|
| Python | 3.10+ | Runtime |
| streamlit | 1.58+ | Web UI |
| pmdarima | 2.1+ | Auto-ARIMA model |
| prophet | 1.3+ | Facebook Prophet model |
| statsmodels | 0.14+ | STS (Structural) model |
| plotly | 6.7+ | Interactive charts |
| pandas | 2.3+ | Data tables & CSV |
| numpy | 2.2+ | Numerical computation |

Install all dependencies at once:

```bash
pip install streamlit pmdarima prophet statsmodels plotly pandas numpy
```

> Prophet and statsmodels are optional — the app runs with only pmdarima installed, and falls back to a log-linear trend if pmdarima is also missing.

---

## Launch

```bash
cd /path/to/ev_scenario_analysis
streamlit run ev_streamlit_app.py
```

The browser opens automatically at **`http://localhost:8501`**.  
To use a different port:

```bash
streamlit run ev_streamlit_app.py --server.port 8502
```

---

## App Structure

```
┌──────────────────────────────────────────────────────────────────┐
│  Sidebar (Settings)           │  Main Area                        │
│                               │                                    │
│  📊 Observed EV Stock         │  ┌───────────────────────────┐    │
│  🔧 Model Settings            │  │ 📈 Forecast               │    │
│     • Forecast Model          │  │ 🔍 Backtest               │    │
│     • Exogenous variables     │  │ 🌐 Scenario Comparison    │    │
│     • Backtest split year     │  │ 💾 Data & Export          │    │
│  🌐 Main Scenario             │  └───────────────────────────┘    │
│     └─ Customize parameters   │                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Sidebar Settings

### 📊 Observed EV Stock (units)

Enter the actual number of registered EVs for each year from 2019 to 2024.  
These values are the training data for all models.

| Field | Default | Notes |
|-------|---------|-------|
| 2019 | 100 | Earliest observed year |
| 2020 | 500 | |
| 2021 | 1,500 | |
| 2022 | 3,500 | |
| 2023 | 5,000 | |
| 2024 | 5,000 | Most recent observation |

---

### 🔧 Model Settings

#### Forecast Model

Select the statistical model used for the **2025–2030 forecast**.

| Model | Backend | Scale | Interval shape | Notes |
|-------|---------|-------|---------------|-------|
| **Auto-ARIMA** | pmdarima | Original | Symmetric | `d=1` forced for small-sample stability |
| **Prophet** | prophet | Original | Symmetric | `changepoint_prior_scale=0.3` |
| **STS (Structural)** | statsmodels | Original | Symmetric | Smooth-trend local level model |

> The **Baseline** scenario always uses the Reference model (see below) regardless of this selector.

#### Use exogenous variables

When enabled, the five external variables — subsidy, charging stations, gas price, EV model count, GDP growth rate — are passed to the selected model as additional predictors.  
Disabled automatically when "Fallback" is active.

> **Note on small datasets:** with only 6 training points and 5 regressors, the model may be over-parameterised. If forecasts look unreasonable, try disabling exogenous variables.

#### Backtest: training end year

A slider to choose the training/test split:

```
◀─── Training ───▶◀── Test ──▶
2019  2020  2021 [2022] 2023  2024
                  ↑
             split year
```

The selected model is fitted on data up to the split year, then evaluated against the held-out years.

---

### 🌐 Main Scenario

Choose one of four pre-defined scenarios for the **2025–2030 forecast window**.  
This controls the assumed future values of the five exogenous variables.

| Scenario | Color | Description |
|----------|-------|-------------|
| **Baseline** | Sky blue (solid) | Uses the Reference model — matches `ev_forecast_simulation.py` exactly |
| **Policy Boost** | Green (dashed) | Higher subsidies, rapid charging expansion, rising gas prices |
| **Infra Constraint** | Orange (dashed) | Slow charging growth, reduced subsidies, lower gas prices |
| **Price Drop** | Purple (dashed) | Battery cost collapse; EV model proliferation; high gas prices |

**Customize scenario parameters (2030 value)**  
Expand this panel to override each external variable's assumed 2030 end-value.  
Values for 2025–2029 are linearly interpolated between the scenario's 2025 start and your chosen 2030 end.

---

## Tabs

### 📈 Forecast

The main forecast chart covering 2019–2030.

```
  units
    │                              ╱╲  95% Prediction Interval
    │                          ╱╱╱  ╲╲╲
    │               ●──●──●  ●──●──●  Model forecast (solid/dashed)
    │       ●──●──●╱                   ╲╲
    │  ●──●╱ Observed                   ╲╲──╱╱
    └──────────────────────────────────────── year
       2019        2024 │ 2025           2030
                        │
                Observed │ Forecast
                 Window  │ Window
```

**Chart elements:**

| Element | Style | Description |
|---------|-------|-------------|
| Black line + dots | Solid | Observed EV stock (2019–2024) |
| Sky blue line + dots | Dashed | Main forecast (selected model & scenario) |
| Light cyan band | Fill | 95% prediction interval |
| Medium blue band | Fill | 50% prediction interval |
| Royalblue dotted + diamond | Dotted | Auto-ARIMA Baseline overlay *(Baseline scenario only)* |
| Gray vertical line | Dashed | Boundary: observed / forecast window |

**Auto-ARIMA Baseline overlay** (visible only when Baseline scenario is selected)  
The royalblue dotted line shows the Auto-ARIMA result alongside the Reference Baseline, so you can see the difference between the two approaches at a glance.

**Hover** over any forecast point to see the predicted value and 95% CI.

**Forecast Table** below the chart shows numeric values for all six forecast years (2025–2030).  
When Baseline is selected, an extra **Auto-ARIMA Baseline** column is appended for comparison.

---

### 🔍 Backtest

Evaluates the selected model's accuracy on held-out observed data.

**Accuracy metrics:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | mean \|actual − forecast\| | Average absolute error in units |
| **MAPE** | mean \|actual − forecast\| / actual × 100 | Percentage error — lower is better |
| **RMSE** | √ mean(actual − forecast)² | Penalises large errors more heavily |

**Backtest chart:**

```
  units
    │  ●──●──●──●  ◆──◆  ← Test Actual (diamond)
    │  Training │  ○──○  ← Model Forecast (orange)
    │  Data     │ [shaded = 95% prediction interval]
    └───────────────────── year
       2019  [split]  2024
```

**Error table** — year-by-year forecast vs. actual, with signed error and error rate.

> When the Baseline scenario is selected, the backtest uses Auto-ARIMA (not the Reference model) because the Reference model has no fitted parameters to evaluate.

---

### 🌐 Scenario Comparison

Overlays all four scenarios — plus the Auto-ARIMA Baseline — on a single chart.

```
  units
    │                              ╱  Policy Boost (green)
    │                         ────   Auto-ARIMA Baseline (royalblue dotted)
    │                    ─────       Baseline / ref (sky blue)
    │               ────╲
    │  ●──●──●──●──●╱    ╲───────    Infra Constraint (orange)
    │                     ╲──────    Price Drop (purple)
    └──────────────────────────────── year
       2019    2024  2025         2030
```

**Scenario table** — forecast values for all scenarios and years side-by-side, including an **Auto-ARIMA Baseline** column.

**Exogenous Variable Assumptions** — a 2 × 3 grid of small charts shows how each of the five external variables is assumed to evolve under each scenario.

---

### 💾 Data & Export

**Observed Data table** — the six years of EV stock and external variable values used for training.

**Forecast Data table** — the selected scenario's 2025–2030 forecast, including prediction intervals, assumed external variable values, and the model name.

**Three CSV download buttons:**

| Button | File | Contents |
|--------|------|----------|
| 📥 Observed Data CSV | `ev_observed.csv` | Training data: EV stock + 5 exogenous variables |
| 📥 Forecast CSV (scenario) | `ev_forecast.csv` | Forecast + 95%/50% intervals + exog assumptions for the selected scenario |
| 📥 All Scenarios CSV | `ev_all_scenarios.csv` | Forecasts and 95% intervals for all 4 scenarios + Auto-ARIMA Baseline |

CSV files use **UTF-8 with BOM** encoding — opens correctly in Excel without re-encoding.

---

## Model Details

### Reference Baseline

Used exclusively for the **Baseline** scenario. Reproduces `ev_forecast_simulation.py` exactly.

```
forecast_h  = ev_obs[-1] + 4000 × h          # linear drift, +4,000 units/year
σ_h         = 729 × h^1.5                     # uncertainty fan
CI_95       = forecast_h ± 1.96  × σ_h        # symmetric
CI_50       = forecast_h ± 0.674 × σ_h
```

Where `h = 1 … 6` is the forecast horizon (2025 = 1, 2030 = 6).

| Year | Forecast | 95% Lower | 95% Upper |
|------|---------|-----------|-----------|
| 2025 | 9,000 | 7,571 | 10,429 |
| 2027 | 17,000 | 9,577 | 24,423 |
| 2030 | 29,000 | 7,997 | 50,003 |

---

### Auto-ARIMA

```python
auto_arima(
    ev_raw,                       # original scale (not log)
    d = 1,                        # first-difference for small-sample stability
    X = exog_scaled,              # 5 z-scored external variables (optional)
    seasonal = False,
    information_criterion = 'aic',
    max_p = 2,  max_q = 2,
)
```

- **`d=1` forced** — prevents explosive or collapsing forecasts that arise when Auto-ARIMA fits higher-order models on only 6 data points. With `d=1` and no regressors the model reduces to ARIMA(0,1,0): `forecast = last_value + mean_annual_diff × h`.
- **Original scale** — intervals are symmetric on the original scale and clipped at 0.
- **Exogenous variables** — z-scored before fitting to equalise scale differences.
- **Fallback** — if pmdarima is not installed, uses ARIMA(0,1,0) analytically (mean difference as drift).

Typical 2030 forecast with default data: **~10,880 units** (conservative, based on recent flattening trend).

---

### Prophet

```python
Prophet(
    yearly_seasonality  = False,
    interval_width      = 0.95,
    changepoint_prior_scale = 0.3,   # moderate flexibility
    uncertainty_samples = 300,
)
# Regressors added per exogenous variable with standardize=True
```

- **Original scale** — Prophet on log scale produces explosive forecasts with only 6 annual points. Original scale is stable.
- **Changepoints** — `changepoint_prior_scale=0.3` gives enough flexibility to detect the recent growth slowdown.
- **50% CI** — scaled from the 95% half-width using the normal z-ratio (0.674 / 1.96).
- **Exogenous variables** — added as regressors; Prophet standardises them internally.

Typical 2030 forecast with default data: **~12,300 units**.

---

### STS (Structural Time Series)

```python
UnobservedComponents(
    ev_raw,
    level = 'smooth trend',    # stochastic level + deterministic slope
    exog  = exog_scaled,       # optional z-scored regressors
)
```

- **Smooth trend** — more stable than `local linear trend` with small datasets (fewer variance parameters to estimate).
- **Original scale** — avoids the wide uncertainty bands that arise from log-scale extrapolation.
- **Intervals** — produced analytically by statsmodels `get_forecast().conf_int(alpha)`.
- **Interpretation** — reflects the most recent structural state of the series. With the observed flattening at 5,000 units, the model forecasts a conservative flat trajectory.

Typical 2030 forecast with default data: **~5,000 units** (flat, wide uncertainty).

---

### Comparison of Models on Default Data

| Model | 2025 | 2027 | 2030 | Interval shape | Scale |
|-------|------|------|------|---------------|-------|
| **Reference (Baseline)** | 9,000 | 17,000 | 29,000 | Symmetric `C·h^1.5` | — |
| **Auto-ARIMA** (d=1) | ~5,980 | ~7,940 | ~10,880 | Symmetric | Original |
| **Prophet** | ~6,594 | ~8,875 | ~12,300 | Symmetric | Original |
| **STS** | ~5,000 | ~5,000 | ~5,000 | Symmetric (wide) | Original |

> The three statistical models (Auto-ARIMA, Prophet, STS) give conservative forecasts because the last two observed years are identical (5,000 units), signalling a growth plateau. The Reference Baseline applies a manually calibrated +4,000/yr drift that matches the original `result.jpg`.

---

## File Overview

```
ev_scenario_analysis/
├── ev_streamlit_app.py         # This Streamlit application
├── ev_forecast_simulation.py   # Original standalone matplotlib simulation
├── ev_forecast_result.png      # Output from the matplotlib simulation
├── result.jpg                  # Reference chart
├── EV_FORECAST_APP_GUIDE.md    # This document
└── README.md                   # Project overview and improvement notes
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: pmdarima` | `pip install pmdarima` |
| `ModuleNotFoundError: prophet` | `pip install prophet` |
| `ModuleNotFoundError: statsmodels` | `pip install statsmodels` |
| Prophet not in model list | Install prophet, then restart the app |
| STS not in model list | Install statsmodels, then restart the app |
| Prophet gives extreme forecast | Disable exogenous variables; `changepoint_prior_scale` is already tuned |
| STS forecast is flat | Expected with recent stagnant data — the model captures structural state |
| Auto-ARIMA forecast is very low | Try disabling exogenous variables (over-parameterised with 6 data points) |
| Port already in use | `streamlit run ev_streamlit_app.py --server.port 8502` |
| Charts not rendering | `pip install --upgrade plotly` |
| CSV opens garbled in Excel | File is UTF-8 BOM — use **Data → From Text/CSV** in Excel |
| Pylance import warnings in VS Code | IDE path issue only; the app runs correctly |
