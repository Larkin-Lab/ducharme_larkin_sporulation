# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:22:42 2026

@author: LarkinLab
"""

# -*- coding: utf-8 -*-
"""
Modified from RecoveryAnalysis.py
Plots 24 hpi and 72 hpi recovery metrics side by side on the same subplot
with slight horizontal offset and WT-comparison significance annotations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D

# =========================
# USER SETTINGS
# =========================
DIR_PATH = 'H:/Analysis/In Process/Recovery'

SAVE_FIGS = False
SAVE_FORMAT = "svg"   # "svg", "png", "pdf", etc.
SAVE_DPI = 300

TIMEPOINTS = ["24", "72"]

FILES_BY_TIME = {
    "24": {
        "WT": f"{DIR_PATH}/WT_recovery24.xlsx",
        "ΔPs": f"{DIR_PATH}/dPs_recovery24.xlsx",
        "ΔpgsB": f"{DIR_PATH}/dpgsB_recovery24.xlsx",
        "ΔepsA-O": f"{DIR_PATH}/deps_recovery24.xlsx",
    },
    "72": {
        "WT": f"{DIR_PATH}/WT_recovery72.xlsx",
        "ΔPs": f"{DIR_PATH}/dPs_recovery72.xlsx",
        "ΔpgsB": f"{DIR_PATH}/dpgsB_recovery72.xlsx",
        "ΔepsA-O": f"{DIR_PATH}/deps_recovery72.xlsx",
    }
}

STRAIN_COLORS = {
    "WT": "#000000",
    "ΔPs": "#1f77b4",
    "ΔpgsB": "#ff7f0e",
    "ΔepsA-O": "#2ca02c",
}

strain_order = ["WT", "ΔPs", "ΔpgsB", "ΔepsA-O"]

phase_order = ["low_1", "high_1", "low_2", "high_2", "low_3"]
baseline_pts = 5

# subplot layout
NROWS = 1
NCOLS = 3

# slight horizontal offset for timepoints
TIME_OFFSETS = {
    "24": -0.14,
    "72":  0.14,
}

# marker shapes for timepoints so the two times are visually distinct
TIME_MARKERS = {
    "24": "o",
    "72": "s",
}

# vertical multiplier for significance text placement
STAR_HEIGHT_MULT = {
    "24": 1.25,
    "72": 1.55,
}

metric_names = [
    "Initial_Damage", "Recovery_Fraction", "Memory_Index"
]

# =========================
# HELPERS
# =========================
def mean_ci(x, alpha=0.05):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)

    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(np.mean(x)), 0.0

    mean = np.mean(x)
    sem = stats.sem(x, nan_policy="omit")
    ci = sem * stats.t.ppf(1 - alpha/2, n - 1)
    return float(mean), float(ci)

def phase_mean(df, col, n=3):
    num_points = len(df[col])
    return df[col].iloc[num_points-n:].mean()

def phase_slope(df, col):
    return np.polyfit(df["time_s"], df[col], 1)[0]

def star(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""

def safe_welch_p(x1, x2):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]

    if len(x1) < 2 or len(x2) < 2:
        return np.nan
    return stats.ttest_ind(x1, x2, equal_var=False).pvalue

# =========================
# LOAD DATA
# =========================
# strain_data[time][strain] = list of replicate phase dicts
strain_data = {}

for time in TIMEPOINTS:
    strain_data[time] = {}
    for strain, file in FILES_BY_TIME[time].items():
        xls = pd.ExcelFile(file)
        reps = []
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            phases = {
                p: df[df["strain_state"] == p].reset_index(drop=True)
                for p in phase_order
            }
            reps.append(phases)
        strain_data[time][strain] = reps

# =========================
# METRIC CALCULATION
# =========================
metrics = []

for time in TIMEPOINTS:
    for strain, reps in strain_data[time].items():
        for r, phases in enumerate(reps):
            low1, high1, low2, high2, low3 = [phases[p] for p in phase_order]

            Gp_l1 = phase_mean(low1, "Gp", baseline_pts)
            Gp_h1 = phase_mean(high1, "Gp", baseline_pts)
            Gp_l2 = phase_mean(low2, "Gp", baseline_pts)
            Gp_h2 = phase_mean(high2, "Gp", baseline_pts)
            Gp_l3 = phase_mean(low3, "Gp", baseline_pts)

            Gpp_h1 = phase_mean(high1, "Gpp", baseline_pts)
            Gpp_l2 = phase_mean(low2, "Gpp", baseline_pts)

            metrics.append({
                "time": time,
                "strain": strain,
                "rep": r,
                "Initial_Damage": 1 - (Gp_h1 / Gp_l1),
                "Recovery_Fraction": (Gp_l2 - Gp_h1) / (Gp_l1 - Gp_h1),
                "Memory_Index": Gp_l3 / Gp_l1
            })

metrics_df = pd.DataFrame(metrics)

# =========================
# SAVE METRIC MEAN + CI TO EXCEL
# =========================
for time in TIMEPOINTS:
    for strain in strain_order:
        metric_summary_rows = []

        for metric in metric_names:
            vals = metrics_df.loc[
                (metrics_df["time"] == time) & (metrics_df["strain"] == strain),
                metric
            ].to_numpy(dtype=float)

            m, ci = mean_ci(vals)

            metric_summary_rows.append({
                "metric": metric,
                "mean": m,
                "ci": ci,
                "n": len(vals)
            })

        metric_summary_df = pd.DataFrame(metric_summary_rows)
        out_file = f"{DIR_PATH}/{strain}_{time}_mean_ci.xlsx"
        metric_summary_df.to_excel(out_file, index=False)

# =========================
# PLOTTING
# =========================
fig, axs = plt.subplots(NROWS, NCOLS, figsize=(5*NCOLS, 4*NROWS))
axs = axs.flatten()

for i, metric in enumerate(metric_names):
    ax = axs[i]

    max_y = -np.inf
    min_y = np.inf

    # First pass: get y limits
    for j, strain in enumerate(strain_order):
        for time in TIMEPOINTS:
            vals = metrics_df.loc[
                (metrics_df["time"] == time) & (metrics_df["strain"] == strain),
                metric
            ].to_numpy(dtype=float)

            m, ci = mean_ci(vals)

            if np.isfinite(m) and np.isfinite(ci):
                max_y = max(max_y, m + 1*ci)
                min_y = min(min_y, m - 1*ci)

    # fallback if all NaN
    if not np.isfinite(max_y):
        max_y = 1.0
    if not np.isfinite(min_y):
        min_y = 0.0

    # Plot means/CI and significance vs WT at same timepoint
    for j, strain in enumerate(strain_order):
        
        # store values for both timepoints
        x_vals = []
        y_vals = []
        
        for time in TIMEPOINTS:
            vals = metrics_df.loc[
                (metrics_df["time"] == time) & (metrics_df["strain"] == strain),
                metric
            ].to_numpy(dtype=float)

            m, ci = mean_ci(vals)
            x = j + TIME_OFFSETS[time]
            
            x_vals.append(x)
            y_vals.append(m)

            ax.errorbar(
                x, m, yerr=ci,
                fmt=TIME_MARKERS[time],
                color=STRAIN_COLORS[strain],
                capsize=5,
                markersize=7,
                linestyle=""
            )
            
            if len(x_vals) == 2 and np.all(np.isfinite(y_vals)):
                ax.plot(
                    x_vals,
                    y_vals,
                    linestyle=":",
                    linewidth=1.5,
                    color=STRAIN_COLORS[strain],
                    alpha=0.8
                )

            if strain != "WT":
                wt_vals = metrics_df.loc[
                    (metrics_df["time"] == time) & (metrics_df["strain"] == "WT"),
                    metric
                ].to_numpy(dtype=float)

                p = safe_welch_p(vals, wt_vals)
                s = star(p)

                if s:
                    # Put 24 h and 72 h significance at different heights
                    if np.isfinite(ci):
                        y = m + STAR_HEIGHT_MULT[time] * ci
                    else:
                        y = m

                    # if ci is zero, still nudge upward a little
                    if ci == 0 or not np.isfinite(ci):
                        y += 0.03 * (max_y - min_y)

                    ax.text(
                        x, y, s,
                        ha="center", va="bottom",
                        fontsize=13
                    )

    ax.set_title(metric.replace("_", " "), fontsize=18)
    ax.set_xticks(np.arange(len(strain_order)))
    ax.set_xticklabels(strain_order, rotation=0, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    i = 0
    for label in ax.get_xticklabels():
        label.set_color(STRAIN_COLORS[strain_order[i]])
        i += 1


    if min_y == max_y:
        pad = 0.1 if min_y == 0 else abs(min_y) * 0.1
        ax.set_ylim(min_y - pad, max_y + pad)
    else:
        pad = 0.4 * (max_y - min_y)
        ax.set_ylim(min_y - pad, max_y + pad)
        
    # Metric-specific y-limits retained from original style
    # if metric == 'initial_damage':
    #     ax.set_ylim(0.97, 1.005)
    # elif metric == 'recovery_fraction':
    #     ax.set_ylim(0, 1)
    # elif metric == 'memory_index':
    #     ax.set_ylim(0, 1)


    # Add a small note so the horizontal offsets/stars are easy to interpret
    # ax.text(
    #     0.02, 0.98,
    #     "○ 24 h   □ 72 h",
    #     transform=ax.transAxes,
    #     ha="left", va="top",
    #     fontsize=11
    # )

# remove unused axes if any
for k in range(len(metric_names), len(axs)):
    fig.delaxes(axs[k])

# -----------------------------
# Legends
# -----------------------------
# strain_handles = [
#     Line2D([0], [0], marker="o", linestyle="", color=STRAIN_COLORS[s], label=s)
#     for s in strain_order
# ]

time_handles = [
    Line2D([0], [0], marker=TIME_MARKERS[t], linestyle="", color="black", label=f"{t} hpi")
    for t in TIMEPOINTS
]

# fig.legend(
#     handles=strain_handles,
#     loc="center left",
#     bbox_to_anchor=(0.83, 0.62),
#     frameon=False,
#     fontsize=16,
#     title="Strain",
#     title_fontsize=16
# )

fig.legend(
    handles=time_handles,
    loc="center left",
    bbox_to_anchor=(0.83, 0.50),
    frameon=False,
    fontsize=16,
    # title="Time",
    title_fontsize=16
)

plt.tight_layout(rect=[0, 0, 0.86, 1])

if SAVE_FIGS:
    out_path = f"{DIR_PATH}/recovery_metrics_24_72_side_by_side.{SAVE_FORMAT}"
    fig.savefig(out_path, bbox_inches="tight", dpi=SAVE_DPI)

plt.show()
plt.close(fig)