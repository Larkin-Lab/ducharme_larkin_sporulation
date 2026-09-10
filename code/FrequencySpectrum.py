# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:33:50 2026

@author: LarkinLab
"""
# -*- coding: utf-8 -*-
"""
Combined Gi vs tau_i plots for all strains at 24 hpi and 72 hpi
using 3-mode generalized Maxwell fits with fixed taus.

Reads one Excel workbook per strain/timepoint.
Each sheet in each workbook is treated as one replicate.
Fits each replicate separately, then summarizes G1, G2, G3 across replicates.

Also computes Welch t-tests comparing each non-WT strain to WT
for each mode (G1, G2, G3) at each timepoint, and adds stars:
    *   p < 0.05
    **  p < 0.01
    *** p < 0.001
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.stats import t, ttest_ind
from matplotlib.ticker import NullLocator


# ============================================================
# USER SETTINGS
# ============================================================

# Root folder containing folders like WT_24, WT_72, dPs_24, etc.
BASE_DIR = r"H:\Analysis\In process\Frequency"

# Strains and timepoints to include
STRAINS = ["WT", "dPs", "dpgsb", "deps"]
STRAIN_NAMES = {"WT": "WT", 
                "dPs": "ΔPs", 
                "dpgsb": "ΔpgsB", 
                "deps": "Δeps"}
TIMES = ["24", "72"]

# Fixed relaxation times (seconds)
# Keep these in ascending order for plotting left -> right
TAUS = np.array([0.01, 0.1, 1.0], dtype=float)

# Colors for each strain
STRAIN_COLORS = {
    "WT": "black",
    "dPs": "#1f77b4",
    "dpgsb": "#ff7f0e",
    "deps": "#2ca02c"
}

normalize = True

# Plot appearance
axis_label_fontsize = 16
tick_label_fontsize = 16
legend_fontsize = 12
title_fontsize = 16
marker_size = 7
capsize = 4
elinewidth = 1.5

# Asterisk appearance
STAR_FONT_SIZE = 14
STAR_Y_PAD_FRAC = 0.035  # fraction of full y-range added above error bar

# Bounds for [G_inf, G1, G2, G3]
LOWER_BOUNDS = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
UPPER_BOUNDS = np.array([np.inf, np.inf, np.inf, np.inf], dtype=float)

# Output directory
OUTPUT_DIR = os.path.join(BASE_DIR, "combined_Gi_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# If None, y-limits are computed automatically from all strains/timepoints
# If you want manual limits, set for example:
# GI_YLIM = (-0.0001, 0.00155)
GI_YLIM = None

# Small horizontal offsets in log10 space, one per strain
# Applied multiplicatively: tau_shifted = tau * 10**offset
LOG10_OFFSETS = {
    "WT":    -0.24,
    "dPs":   -0.08,
    "dpgsb":  0.08,
    "deps":   0.24
}


# ============================================================
# MODEL FUNCTIONS
# ============================================================

def generalized_maxwell_gp_gpp(omega, params, taus):
    """
    3-mode generalized Maxwell model with fixed taus and fitted params:
    params = [G_inf, G1, G2, G3]

    G'(w)  = G_inf + sum_i G_i * (w^2 tau_i^2)/(1 + w^2 tau_i^2)
    G''(w) =         sum_i G_i * (w tau_i)/(1 + w^2 tau_i^2)
    """
    G_inf = params[0]
    Gs = np.array(params[1:], dtype=float)

    omega = np.asarray(omega, dtype=float)
    gp = np.full_like(omega, G_inf, dtype=float)
    gpp = np.zeros_like(omega, dtype=float)

    for Gi, taui in zip(Gs, taus):
        wt = omega * taui
        gp += Gi * (wt**2) / (1.0 + wt**2)
        gpp += Gi * wt / (1.0 + wt**2)

    return gp, gpp


def residuals(params, omega, gp_obs, gpp_obs, taus, mode="relative"):
    gp_fit, gpp_fit = generalized_maxwell_gp_gpp(omega, params, taus)

    if mode == "relative":
        eps = 1e-15
        r_gp = (gp_fit - gp_obs) / np.maximum(np.abs(gp_obs), eps)
        r_gpp = (gpp_fit - gpp_obs) / np.maximum(np.abs(gpp_obs), eps)
    else:
        r_gp = gp_fit - gp_obs
        r_gpp = gpp_fit - gpp_obs

    return np.concatenate([r_gp, r_gpp])


def initial_guess_from_data(omega, gp, gpp):
    gp = np.asarray(gp, dtype=float)
    gpp = np.asarray(gpp, dtype=float)

    G_inf0 = max(np.min(gp) * 0.5, 0.0)
    span = max(np.max(gp) - G_inf0, np.max(gpp), 1e-12)

    G1_0 = 0.5 * span
    G2_0 = 0.3 * span
    G3_0 = 0.2 * span

    return np.array([G_inf0, G1_0, G2_0, G3_0], dtype=float)


def fit_single_replicate(omega, gp, gpp, taus):
    x0 = initial_guess_from_data(omega, gp, gpp)

    result = least_squares(
        residuals,
        x0=x0,
        bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
        args=(omega, gp, gpp, taus, "relative"),
        method="trf",
        max_nfev=20000
    )

    params = result.x
    gp_fit, gpp_fit = generalized_maxwell_gp_gpp(omega, params, taus)

    return {
        "params": params,
        "gp_fit": gp_fit,
        "gpp_fit": gpp_fit,
        "cost": result.cost,
        "success": result.success,
        "message": result.message,
        "nfev": result.nfev
    }


# ============================================================
# STATS HELPERS
# ============================================================

def mean_ci_95(arr, axis=0):
    arr = np.asarray(arr, dtype=float)
    n = arr.shape[axis]
    mean = np.mean(arr, axis=axis)

    if n <= 1:
        ci = np.zeros_like(mean)
        return mean, ci

    sd = np.std(arr, axis=axis, ddof=1)
    sem = sd / np.sqrt(n)
    tcrit = t.ppf(0.975, df=n - 1)
    ci = tcrit * sem
    return mean, ci


def p_to_stars(p):
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


# ============================================================
# FILE / DATA HELPERS
# ============================================================

def get_input_file(base_dir, strain, time):
    folder = os.path.join(base_dir, f"{strain}_{time}")
    filename = f"{strain}_frequency_sweep{time}.xlsx"
    return os.path.join(folder, filename)


def read_workbook_replicates(input_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Could not find input file:\n{input_file}")

    excel_file = pd.ExcelFile(input_file)
    sheet_names = excel_file.sheet_names

    if len(sheet_names) == 0:
        raise ValueError(f"No sheets found in workbook:\n{input_file}")

    replicate_data = []

    for sheet in sheet_names:
        df = pd.read_excel(input_file, sheet_name=sheet)

        expected_cols = {"omega", "Gp", "Gpp"}
        if not expected_cols.issubset(df.columns):
            raise ValueError(
                f"Sheet '{sheet}' in '{os.path.basename(input_file)}' "
                f"must contain columns {expected_cols}, but has {list(df.columns)}"
            )

        df = df[["omega", "Gp", "Gpp"]].dropna().copy()

        replicate_data.append({
            "sheet": sheet,
            "omega": df["omega"].to_numpy(dtype=float),
            "Gp": df["Gp"].to_numpy(dtype=float),
            "Gpp": df["Gpp"].to_numpy(dtype=float)
        })

    return replicate_data


def summarize_one_condition(base_dir, strain, time, taus):
    """
    For one strain/timepoint workbook:
    - read replicate sheets
    - fit each replicate
    - return summary stats for G_inf, G1, G2, G3
    """
    input_file = get_input_file(base_dir, strain, time)
    replicate_data = read_workbook_replicates(input_file)

    fit_results = []
    for rep in replicate_data:
        fit_out = fit_single_replicate(rep["omega"], rep["Gp"], rep["Gpp"], taus)
        fit_results.append({
            "sheet": rep["sheet"],
            **fit_out
        })

    param_matrix = np.vstack([fr["params"] for fr in fit_results])  # columns: G_inf, G1, G2, G3
    # normalize by G_total
    if normalize:
        for i in np.arange(3):
            param_matrix[i] /= sum(param_matrix[i])
    param_means, param_cis = mean_ci_95(param_matrix, axis=0)

    result = {
        "strain": strain,
        "time": time,
        "n_replicates": len(fit_results),
        "input_file": input_file,
        "G_inf_mean": param_means[0],
        "G_inf_ci95": param_cis[0],
        "G1_mean": param_means[1],
        "G1_ci95": param_cis[1],
        "G2_mean": param_means[2],
        "G2_ci95": param_cis[2],
        "G3_mean": param_means[3],
        "G3_ci95": param_cis[3],
        "tau1": taus[0],
        "tau2": taus[1],
        "tau3": taus[2]
    }

    return result, fit_results


# ============================================================
# P-VALUE HELPERS
# ============================================================

def compute_wt_pvalues(all_fit_results, strains, times):
    """
    Compare each non-WT strain to WT for G1, G2, G3 at each timepoint.
    Uses replicate-level fitted parameters.

    Returns nested dict:
    pvals[time][strain][mode_index] = raw p-value
    where mode_index = 0,1,2 corresponds to G1,G2,G3
    """
    pvals = {time: {} for time in times}

    for time in times:
        wt_params = np.vstack([fr["params"] for fr in all_fit_results[("WT", time)]])
        wt_gis = wt_params[:, 0:4]   # Ginf, G1, G2, G3

        for strain in strains:
            if strain == "WT":
                continue

            strain_params = np.vstack([fr["params"] for fr in all_fit_results[(strain, time)]])
            strain_gis = strain_params[:, 0:4]

            pvals[time][strain] = {}

            for mode_idx in range(4):
                wt_vals = wt_gis[:, mode_idx]
                strain_vals = strain_gis[:, mode_idx]

                if len(wt_vals) < 2 or len(strain_vals) < 2:
                    pvals[time][strain][mode_idx] = np.nan
                else:
                    stat = ttest_ind(
                        wt_vals,
                        strain_vals,
                        equal_var=False,
                        nan_policy="omit"
                    )
                    pvals[time][strain][mode_idx] = stat.pvalue
    print(pvals)
    return pvals


# ============================================================
# COLLECT RESULTS FOR ALL STRAINS / TIMES
# ============================================================

summary_rows = []
all_fit_results = {}

for strain in STRAINS:
    for time in TIMES:
        summary, fit_results = summarize_one_condition(BASE_DIR, strain, time, TAUS)
        summary_rows.append(summary)
        all_fit_results[(strain, time)] = fit_results

summary_df = pd.DataFrame(summary_rows)

# Compute WT comparison p-values from replicate-level Gi fits
pvals = compute_wt_pvalues(all_fit_results, STRAINS, TIMES)

# Save overall summary and p-values
summary_excel = os.path.join(OUTPUT_DIR, "all_strains_all_times_gmaxwell3_Gi_summary.xlsx")

pval_rows = []
for time in TIMES:
    for strain in STRAINS:
        if strain == "WT":
            continue
        mode_names = ["G_inf", "G1", "G2", "G3"]
        taus_out = [np.inf, *TAUS]

        for mode_idx, (mode_name, tau) in enumerate(zip(mode_names, taus_out)):
            raw_p = pvals[time][strain][mode_idx]
            pval_rows.append({
                "time": time,
                "strain": strain,
                "comparison": f"{strain} vs WT",
                "mode": mode_name,
                "tau": tau,
                "raw_pvalue": raw_p,
                "stars": p_to_stars(raw_p)
            })

pvals_df = pd.DataFrame(pval_rows)

with pd.ExcelWriter(summary_excel, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Gi_summary", index=False)
    pvals_df.to_excel(writer, sheet_name="WT_vs_WT_pvalues", index=False)


# ============================================================
# DETERMINE COMMON Y-LIMITS
# ============================================================

if GI_YLIM is None:
    y_min_candidates = []
    y_max_candidates = []

    for _, row in summary_df.iterrows():
        means = np.array([row["G1_mean"], row["G2_mean"], row["G3_mean"]], dtype=float)
        cis = np.array([row["G1_ci95"], row["G2_ci95"], row["G3_ci95"]], dtype=float)

        y_min_candidates.extend(list(means - cis))
        y_max_candidates.extend(list(means + cis))

    y_min = min(y_min_candidates)
    y_max = max(y_max_candidates)

    data_range = y_max - y_min
    if data_range <= 0:
        data_range = max(abs(y_max), 1.0)

    bottom_pad = 0.05 * data_range
    top_pad = 0.14 * data_range  # extra room for stars

    y_low = min(0.0, y_min - bottom_pad)
    y_high = y_max + top_pad
    COMMON_YLIM = (y_low, y_high)
else:
    COMMON_YLIM = GI_YLIM


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_all_strains_one_time(summary_df, time, taus, colors, output_dir, common_ylim, pvals):
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(6,5.5),
        sharey=True,
        gridspec_kw={
            "width_ratios":[4,1],
            "wspace":0.05
        }
    )
    
    # Hide touching spines
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    
    ax_left.tick_params(right=False)
    ax_right.tick_params(left=False)
    
    # # Left axis (right edge)
    # kwargs = dict(transform=ax_left.transAxes, color="k", clip_on=False)
    
    # # Right axis (left edge)
    # kwargs = dict(transform=ax_right.transAxes, color="k", clip_on=False)
    

    y_range = common_ylim[1] - common_ylim[0]
    star_pad = STAR_Y_PAD_FRAC * y_range

    for strain in STRAINS:
        row = summary_df[(summary_df["strain"] == strain) & (summary_df["time"] == time)]
        if row.empty:
            continue
        row = row.iloc[0]

        gi_means = np.array([
            row["G_inf_mean"],
            row["G1_mean"],
            row["G2_mean"],
            row["G3_mean"]
        ], dtype=float)
        
        gi_cis = np.array([
            row["G_inf_ci95"],
            row["G1_ci95"],
            row["G2_ci95"],
            row["G3_ci95"]
        ], dtype=float)
        
        # left axis (G_inf)
        x = taus * (10**LOG10_OFFSETS[strain])

        ax_left.errorbar(
            x,
            gi_means[1:],
            yerr=gi_cis[1:],
            fmt="o",
            linestyle="None",
            markersize=marker_size,
            capsize=capsize,
            elinewidth=elinewidth,
            color=colors[strain],
            label=STRAIN_NAMES[strain]
        )
        
        GINF_X = 20.0        # pseudo-position beyond τ = 1 s

        x_inf = GINF_X * (10**(2.3*LOG10_OFFSETS[strain]))

        ax_right.errorbar(
            x_inf,
            gi_means[0],
            yerr=gi_cis[0],
            fmt="o",
            linestyle="None",
            markersize=marker_size,
            capsize=capsize,
            elinewidth=elinewidth,
            color=colors[strain],
        )
        
        if strain != "WT":

            raw_p = pvals[time][strain][0]
            stars = p_to_stars(raw_p)
        
            if stars != "":
                y_star = gi_means[0] + gi_cis[0] + star_pad
        
                ax_right.text(
                    x_inf,
                    y_star,
                    stars,
                    ha="center",
                    va="bottom",
                    fontsize=STAR_FONT_SIZE,
                    color="black",
                )

        # Add significance stars for non-WT strains
        if strain != "WT":
            for mode_idx in range(1,4):
                raw_p = pvals[time][strain][mode_idx]

                stars = p_to_stars(raw_p)
                
                if stars != "":
                    y_star = gi_means[mode_idx] + gi_cis[mode_idx] + star_pad
                
                    ax_left.text(
                        x[mode_idx-1],
                        y_star,
                        stars,
                        ha="center",
                        va="bottom",
                        fontsize=STAR_FONT_SIZE,
                        color="black",
                    )


    ax_left.set_xscale("log")
    ax_left.set_xlim(3e-3,5)
    ax_right.set_xscale('log')
    ax_right.set_xlim(2,200)

    ax_left.set_xticks([0.01, 0.1, 1.0], minor=False)
    ax_left.set_xticklabels(["0.01", "0.1", "1"])
    ax_right.set_xticks([20.0], minor=False)
    ax_right.set_xticklabels([r"$\infty$"])

    ax_left.set_ylim(0,0.7)
    ax_right.set_ylim(0,0.7)

    # ax_left.set_xlabel(r"Fixed relaxation time $\tau_i$ (s)", fontsize=axis_label_fontsize)
    ax_left.set_ylabel(r"Norm. mode modulus $G_i$ (a.u.)", fontsize=axis_label_fontsize)
    # ax_left.set_title(f"{time} hours", fontsize=title_fontsize)

    ax_left.tick_params(axis="both", labelsize=tick_label_fontsize)
    ax_right.tick_params(axis='x', labelsize=tick_label_fontsize+8)
    
    # Remove all minor ticks from log axes
    ax_left.xaxis.set_minor_locator(NullLocator())
    ax_right.xaxis.set_minor_locator(NullLocator())

    
    # ax.grid(False, which="both", alpha=0.3)
    ax_left.legend(fontsize=legend_fontsize, frameon=False, loc='upper right')
    
    plt.tight_layout()

    # ----- axis break -----
    from matplotlib.lines import Line2D
    
    bbox1 = ax_left.get_position()
    bbox2 = ax_right.get_position()
    
    dx = 0.006
    dy = 0.010
    
    # Bottom
    fig.add_artist(Line2D(
        [bbox1.x1-dx, bbox1.x1+dx],
        [bbox1.y0-dy, bbox1.y0+dy],
        transform=fig.transFigure, color='k', lw=1.2))
    
    fig.add_artist(Line2D(
        [bbox2.x0-dx, bbox2.x0+dx],
        [bbox2.y0-dy, bbox2.y0+dy],
        transform=fig.transFigure, color='k', lw=1.2))
    
    # Top
    fig.add_artist(Line2D(
        [bbox1.x1-dx, bbox1.x1+dx],
        [bbox1.y1-dy, bbox1.y1+dy],
        transform=fig.transFigure, color='k', lw=1.2))
    
    fig.add_artist(Line2D(
        [bbox2.x0-dx, bbox2.x0+dx],
        [bbox2.y1-dy, bbox2.y1+dy],
        transform=fig.transFigure, color='k', lw=1.2))

    outpath = os.path.join(output_dir, f"all_strains_{time}h_Gi_vs_tau_ci95.svg")
    fig.suptitle(f"{time} hours", fontsize=title_fontsize, y=0.93)
    fig.supxlabel(r"Fixed relaxation time $\tau_i$ (s)", fontsize=axis_label_fontsize)
    fig.savefig(outpath, format="svg", bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return outpath


def plot_two_panel(summary_df, times, taus, colors, output_dir, common_ylim, pvals):

    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator

    # ============================================================
    # CREATE TWO BROKEN-AXIS PLOTS SIDE BY SIDE
    # ============================================================

    fig = plt.figure(figsize=(15.0, 5.5))

    # Each timepoint gets the exact same 4:1 broken-axis structure
    # as plot_all_strains_one_time()
    outer_gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1, 1],
        wspace=0.18
    )

    axes = []

    for col, time in enumerate(times):

        # --------------------------------------------------------
        # Exact same subplot structure as single plot
        # --------------------------------------------------------

        gs = outer_gs[col].subgridspec(
            1,
            2,
            width_ratios=[4, 1],
            wspace=0.05
        )

        ax_left = fig.add_subplot(gs[0])
        ax_right = fig.add_subplot(gs[1], sharey=ax_left)

        axes.append((ax_left, ax_right))

        # --------------------------------------------------------
        # Hide touching spines
        # --------------------------------------------------------

        ax_left.spines["right"].set_visible(False)
        ax_right.spines["left"].set_visible(False)

        ax_left.tick_params(right=False)
        ax_right.tick_params(left=False)

        d = 0.015

        # --------------------------------------------------------
        # Plot data
        # --------------------------------------------------------

        y_range = common_ylim[1] - common_ylim[0]
        star_pad = STAR_Y_PAD_FRAC * y_range

        for strain in STRAINS:

            row = summary_df[
                (summary_df["strain"] == strain) &
                (summary_df["time"] == time)
            ]

            if row.empty:
                continue

            row = row.iloc[0]

            gi_means = np.array([
                row["G_inf_mean"],
                row["G1_mean"],
                row["G2_mean"],
                row["G3_mean"]
            ], dtype=float)

            gi_cis = np.array([
                row["G_inf_ci95"],
                row["G1_ci95"],
                row["G2_ci95"],
                row["G3_ci95"]
            ], dtype=float)

            # ----------------------------------------------------
            # LEFT AXIS: G1, G2, G3
            # ----------------------------------------------------

            x = taus * (10 ** LOG10_OFFSETS[strain])

            ax_left.errorbar(
                x,
                gi_means[1:],
                yerr=gi_cis[1:],
                fmt="o",
                linestyle="None",
                markersize=marker_size,
                capsize=capsize,
                elinewidth=elinewidth,
                color=colors[strain],
                label=STRAIN_NAMES[strain]
            )

            # ----------------------------------------------------
            # RIGHT AXIS: G_inf
            # ----------------------------------------------------

            GINF_X = 20.0

            x_inf = GINF_X * (10 ** LOG10_OFFSETS[strain])

            ax_right.errorbar(
                x_inf,
                gi_means[0],
                yerr=gi_cis[0],
                fmt="o",
                linestyle="None",
                markersize=marker_size,
                capsize=capsize,
                elinewidth=elinewidth,
                color=colors[strain],
            )

            # ----------------------------------------------------
            # SIGNIFICANCE STAR FOR G_inf
            # ----------------------------------------------------

            if strain != "WT":

                raw_p = pvals[time][strain][0]
                stars = p_to_stars(raw_p)

                if stars != "":

                    y_star = (
                        gi_means[0]
                        + gi_cis[0]
                        + star_pad
                    )

                    ax_right.text(
                        x_inf,
                        y_star,
                        stars,
                        ha="center",
                        va="bottom",
                        fontsize=STAR_FONT_SIZE,
                        color="black",
                    )

            # ----------------------------------------------------
            # SIGNIFICANCE STARS FOR G1-G3
            # ----------------------------------------------------

            if strain != "WT":

                for mode_idx in range(1, 4):

                    raw_p = pvals[time][strain][mode_idx]
                    stars = p_to_stars(raw_p)

                    if stars != "":

                        y_star = (
                            gi_means[mode_idx]
                            + gi_cis[mode_idx]
                            + star_pad
                        )

                        ax_left.text(
                            x[mode_idx - 1],
                            y_star,
                            stars,
                            ha="center",
                            va="bottom",
                            fontsize=STAR_FONT_SIZE,
                            color="black",
                        )

        # ========================================================
        # X AXES
        # ========================================================

        ax_left.set_xscale("log")
        ax_right.set_xscale("log")

        ax_left.set_xlim(2e-3, 5)
        ax_right.set_xlim(2e-1, 3.5e1)

        # Only manually specified ticks
        ax_left.set_xticks(
            [0.01, 0.1, 1.0],
            minor=False
        )

        ax_left.set_xticklabels(
            ["0.01", "0.1", "1"]
        )

        ax_right.set_xticks(
            [20.0],
            minor=False
        )

        ax_right.set_xticklabels(
            [r"$\infty$"]
        )

        # Remove automatically generated minor ticks
        ax_left.xaxis.set_minor_locator(NullLocator())
        ax_right.xaxis.set_minor_locator(NullLocator())

        # ========================================================
        # Y AXES
        # ========================================================

        ax_left.set_ylim(common_ylim)
        ax_right.set_ylim(common_ylim)

        # Remove y ticks from right-hand broken-axis section
        ax_left.yaxis.set_ticks_position("left")
        ax_right.yaxis.set_ticks_position("none")

        # ========================================================
        # LABELS / TITLE
        # ========================================================

        ax_left.set_xlabel(
            r"Fixed relaxation time $\tau_i$ (s)",
            fontsize=axis_label_fontsize
        )

        # Match the single-plot title
        ax_left.set_title(
            f"{time} hours: $G_i$ vs $\\tau_i$ (95% CI)",
            fontsize=title_fontsize
        )

        # Only first plot gets the y-axis label
        if col == 0:
            ax_left.set_ylabel(
                r"Norm. mode modulus $G_i$ (a.u.)",
                fontsize=axis_label_fontsize
            )

        # ========================================================
        # TICK LABEL SIZES
        # ========================================================

        ax_left.tick_params(
            axis="both",
            labelsize=tick_label_fontsize
        )

        ax_right.tick_params(
            axis="x",
            labelsize=tick_label_fontsize
        )

        # ========================================================
        # LEGEND
        # ========================================================

        # Put legend on the right-hand timepoint, exactly as in
        # the single plot
        if col == 1:
            ax_left.legend(
                fontsize=legend_fontsize,
                frameon=False,
                loc="upper right"
            )

    # ============================================================
    # ADD AXIS-BREAK DIAGONALS
    # ============================================================

    for ax_left, ax_right in axes:

        bbox1 = ax_left.get_position()
        bbox2 = ax_right.get_position()

        dx = 0.006
        dy = 0.010

        # --------------------------------------------------------
        # Bottom-left
        # --------------------------------------------------------

        fig.add_artist(Line2D(
            [bbox1.x1-dx, bbox1.x1+dx],
            [bbox1.y0-dy, bbox1.y0+dy],
            transform=fig.transFigure,
            color='k',
            lw=1.2
        ))

        # --------------------------------------------------------
        # Bottom-right
        # --------------------------------------------------------

        fig.add_artist(Line2D(
            [bbox2.x0-dx, bbox2.x0+dx],
            [bbox2.y0-dy, bbox2.y0+dy],
            transform=fig.transFigure,
            color='k',
            lw=1.2
        ))

        # --------------------------------------------------------
        # Top-left
        # --------------------------------------------------------

        fig.add_artist(Line2D(
            [bbox1.x1-dx, bbox1.x1+dx],
            [bbox1.y1-dy, bbox1.y1+dy],
            transform=fig.transFigure,
            color='k',
            lw=1.2
        ))

        # --------------------------------------------------------
        # Top-right
        # --------------------------------------------------------

        fig.add_artist(Line2D(
            [bbox2.x0-dx, bbox2.x0+dx],
            [bbox2.y1-dy, bbox2.y1+dy],
            transform=fig.transFigure,
            color='k',
            lw=1.2
        ))

    # ============================================================
    # SAVE
    # ============================================================

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.16,
        top=0.90,
        wspace=0.18
    )

    outpath = os.path.join(
        output_dir,
        "all_strains_24h_72h_Gi_vs_tau_ci95_2panel.svg"
    )

    fig.savefig(
        outpath,
        format="svg",
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    return outpath


# ============================================================
# MAKE THE PLOTS
# ============================================================

plot24_out = plot_all_strains_one_time(
    summary_df=summary_df,
    time="24",
    taus=TAUS,
    colors=STRAIN_COLORS,
    output_dir=OUTPUT_DIR,
    common_ylim=COMMON_YLIM,
    pvals=pvals
)

plot72_out = plot_all_strains_one_time(
    summary_df=summary_df,
    time="72",
    taus=TAUS,
    colors=STRAIN_COLORS,
    output_dir=OUTPUT_DIR,
    common_ylim=COMMON_YLIM,
    pvals=pvals
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\nSaved summary Excel:")
print(summary_excel)

print("\nSaved SVG plots:")
print(plot24_out)
print(plot72_out)
# print(combined_out)

print("\nCommon y-limits used for both timepoints:")
print(COMMON_YLIM)

print("\nWT comparison p-values:")
for time in TIMES:
    print(f"\nTime = {time} h")
    for strain in STRAINS:
        if strain == "WT":
            continue
        vals = [pvals[time][strain][i] for i in range(3)]
        stars = [p_to_stars(v) for v in vals]
        print(
            f"{strain} vs WT | "
            f"G1: {vals[0]:.4g} {stars[0]} | "
            f"G2: {vals[1]:.4g} {stars[1]} | "
            f"G3: {vals[2]:.4g} {stars[2]}"
        )