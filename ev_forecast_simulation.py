"""
EV Stock Forecast Simulation
Reproduces the observed EV stock chart with Auto-ARIMA-style baseline forecast,
prediction intervals, and scenario analysis (2019-2030).
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

np.random.seed(42)

# ── Observed data (2019–2024) ──────────────────────────────────────────────
years_obs = np.array([2019, 2020, 2021, 2022, 2023, 2024])
ev_obs    = np.array([100,  500,  1500, 3500, 5000, 5000])   # actual units

# ── ARIMA baseline forecast (2025–2030) ───────────────────────────────────
# ARIMA(0,1,0)-like: last observed value + mean recent difference * h
recent_diffs = np.diff(ev_obs[-3:])          # last 2 year-on-year changes
step         = float(np.mean(recent_diffs))  # ≈ 0  →  anchor to last jump to 2025

# Override: the ARIMA model picks up a ~4 000/yr trend visible in 2022–2024
trend_step = 4000
years_fc   = np.array([2025, 2026, 2027, 2028, 2029, 2030])
h          = years_fc - 2024                 # forecast horizon 1..6
arima_fc   = ev_obs[-1] + trend_step * h    # [9000, 13000, 17000, 21000, 25000, 29000]

# ── Prediction intervals ─────────────────────────────────────────────────
# Uncertainty grows as σ_h = C * h^1.5  (faster-than-random-walk spread)
# Calibrated so 95% upper at h=6 ≈ 50 000
C      = 729.0
sigma_h = C * h ** 1.5

z95 = 1.96
z50 = 0.674

upper95 = arima_fc + z95 * sigma_h
lower95 = np.maximum(0, arima_fc - z95 * sigma_h)

upper50 = arima_fc + z50 * sigma_h
lower50 = np.maximum(0, arima_fc - z50 * sigma_h)

# ── Scenario forecasts ────────────────────────────────────────────────────
# Accelerated: higher adoption rate → ~35 000 by 2030
accel_fc    = ev_obs[-1] + (years_fc - 2024) * 5200   # ~9k→35k

# Conservative: slower uptake → ~14 000 by 2030
conserv_fc  = ev_obs[-1] + (years_fc - 2024) * 1000   # ~9k→14k

# ── Plotting ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 8))

# All x-data combined for convenient limits
all_years = np.concatenate([years_obs, years_fc])

# 95 % prediction interval (light cyan)
ax.fill_between(years_fc, lower95, upper95,
                color='lightcyan', edgecolor='lightcyan',
                alpha=0.8, zorder=1, label='95% prediction interval')

# 50 % prediction interval (sky-blue, darker)
ax.fill_between(years_fc, lower50, upper50,
                color='deepskyblue', edgecolor='deepskyblue',
                alpha=0.5, zorder=2, label='50% prediction interval')

# Observed EV stock (black line + circles)
ax.plot(years_obs, ev_obs,
        color='black', linewidth=2, marker='o', markersize=8,
        zorder=6, label='Observed EV stock')

# Bridge line connecting last observed point (2024) to first forecast point (2025)
ax.plot([years_obs[-1], years_fc[0]], [ev_obs[-1], arima_fc[0]],
        color='black', linewidth=2, zorder=6)

# Accelerated scenario (green dashed)
ax.plot(years_fc, accel_fc,
        color='green', linewidth=2, linestyle='--',
        zorder=5, label='Accelerated scenario')

# Auto-ARIMA baseline forecast (blue dashed + large blue dots)
ax.plot(years_fc, arima_fc,
        color='royalblue', linewidth=2, linestyle='--',
        marker='o', markersize=11,
        markerfacecolor='blue', markeredgecolor='blue',
        zorder=7, label='Auto-ARIMA baseline forecast')

# Conservative scenario (red dotted)
ax.plot(years_fc, conserv_fc,
        color='red', linewidth=2, linestyle=':',
        zorder=5, label='Conservative scenario')

# ── Dividing line at 2025 ─────────────────────────────────────────────────
ax.axvline(x=2025, color='dimgray', linestyle='--', linewidth=2, alpha=0.7, zorder=8)

# ── Annotation text ───────────────────────────────────────────────────────
ax.text(2021.5, 47000, 'Observed evidence window',
        fontsize=14, fontweight='bold', ha='center', va='center')
ax.text(2027.5, 47000, 'Forecast scenario window',
        fontsize=14, fontweight='bold', ha='center', va='center')

# ── Axes formatting ───────────────────────────────────────────────────────
ax.set_xlabel('Year', fontsize=14, labelpad=8)
ax.set_ylabel('Total Registered EV Stock', fontsize=14, labelpad=8)

ax.set_xlim(2018.5, 2030.7)
ax.set_ylim(0, 55000)

ax.set_xticks(range(2019, 2031))
ax.set_xticklabels(range(2019, 2031), fontsize=12)
ax.set_yticks([0, 10000, 20000, 30000, 40000, 50000])
ax.set_yticklabels(['0', '1', '2', '3', '4', '5'], fontsize=12)

# ×10^4 exponent label (top-left of y-axis)
ax.text(-0.065, 1.02, r'$\times10^4$',
        transform=ax.transAxes, fontsize=12, va='bottom')

ax.grid(True, which='both', linestyle=':', linewidth=0.4, color='gray', alpha=0.5)
ax.set_axisbelow(True)

# ── Legend ────────────────────────────────────────────────────────────────
legend_elements = [
    Patch(facecolor='lightcyan',   edgecolor='lightcyan',   alpha=0.8,  label='95% prediction interval'),
    Patch(facecolor='deepskyblue', edgecolor='deepskyblue', alpha=0.6,  label='50% prediction interval'),
    Line2D([0], [0], color='black',     linewidth=2, marker='o', markersize=8,
           label='Observed EV stock'),
    Line2D([0], [0], color='green',     linewidth=2, linestyle='--',
           label='Accelerated scenario'),
    Line2D([0], [0], color='royalblue', linewidth=2, linestyle='--',
           marker='o', markersize=9, markerfacecolor='blue',
           label='Auto-ARIMA baseline forecast'),
    Line2D([0], [0], color='red',       linewidth=2, linestyle=':',
           label='Conservative scenario'),
]

ax.legend(handles=legend_elements,
          loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=3, fontsize=11, frameon=True,
          columnspacing=1.5, handlelength=2.5)

plt.tight_layout()

out_path = 'ev_forecast_result.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved → {out_path}")

# Show interactively when a display is available
try:
    plt.show()
except Exception:
    pass  # headless environment: PNG already saved above
