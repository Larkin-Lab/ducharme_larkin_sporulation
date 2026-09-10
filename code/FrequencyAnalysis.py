# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:57:01 2026

@author: larki
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.stats import t


# ============================================================
# USER SETTINGS
# ============================================================

# Easy to swap:
strain = "WT"          # options: "WT", "dPs", "dpgsb", "deps"
time = "24"            # "24" or "72"


save_fig = False

# Folder containing your Excel files
DATA_DIR = rf"H:\Analysis\In process\Frequency\{strain}_{time}"

# Expected file naming pattern
input_file = os.path.join(DATA_DIR, f"{strain}_frequency_sweep{time}.xlsx")

# Output folder
OUTPUT_DIR = os.path.join(DATA_DIR, "fit_g")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fixed relaxation times (seconds)
TAUS = np.array([1.0, 0.1, 0.01], dtype=float)

# Optional plot settings
axis_label_fontsize = 20
tick_label_fontsize = 18
legend_fontsize = 14
ALPHA = 0.1

# Bounds for [G_inf, G1, G2, G3]
# Set lower bounds >= 0 so moduli stay nonnegative
LOWER_BOUNDS = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
UPPER_BOUNDS = np.array([np.inf, np.inf, np.inf, np.inf], dtype=float)

FIT_COLOR = "#7A7A7A"   # lighter, more subdued

STRAIN_COLORS = {
    "WT": "black",
    "dPs": "#1f77b4",
    "dpgsb": "#ff7f0e",
    "deps": "#2ca02c"
}

# ============================================================
# MODEL FUNCTIONS
# ============================================================

def generalized_maxwell_gp_gpp(omega, params, taus):
    """
    Generalized Maxwell model with fixed taus and fitted parameters:
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
    """
    Residual vector for simultaneous fitting of G' and G''.
    """
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
    """
    Heuristic initial guess for [G_inf, G1, G2, G3].
    """
    gp = np.asarray(gp, dtype=float)
    gpp = np.asarray(gpp, dtype=float)

    G_inf0 = max(np.min(gp) * 0.5, 0.0)
    span = max(np.max(gp) - G_inf0, np.max(gpp), 1e-10)

    # Split span across the three modes
    G1_0 = 0.5 * span
    G2_0 = 0.3 * span
    G3_0 = 0.2 * span

    return np.array([G_inf0, G1_0, G2_0, G3_0], dtype=float)


def fit_single_replicate(omega, gp, gpp, taus):
    """
    Fit one replicate and return fitted params and fitted curves.
    """
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

def mean_sd(arr, axis=0):
    arr = np.asarray(arr, dtype=float)
    mean = np.mean(arr, axis=axis)
    sd = np.std(arr, axis=axis, ddof=1) if arr.shape[axis] > 1 else np.zeros_like(mean)
    return mean, sd


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

def compute_relative_residuals(g_obs, g_fit):
    eps = 1e-15
    return (g_fit - g_obs) / np.maximum(np.abs(g_obs), eps)


# ============================================================
# READ DATA
# ============================================================

if not os.path.exists(input_file):
    raise FileNotFoundError(f"Could not find input file:\n{input_file}")

excel_file = pd.ExcelFile(input_file)
sheet_names = excel_file.sheet_names

if len(sheet_names) == 0:
    raise ValueError("No sheets found in workbook.")

replicate_data = []
replicate_names = []

for sheet in sheet_names:
    df = pd.read_excel(input_file, sheet_name=sheet)

    expected_cols = {"omega", "Gp", "Gpp"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"Sheet '{sheet}' must contain columns {expected_cols}, "
            f"but has columns: {list(df.columns)}"
        )

    df = df[["omega", "Gp", "Gpp"]].dropna().copy()

    omega = df["omega"].to_numpy(dtype=float)
    gp = df["Gp"].to_numpy(dtype=float)
    gpp = df["Gpp"].to_numpy(dtype=float)

    replicate_data.append({
        "sheet": sheet,
        "omega": omega,
        "Gp": gp,
        "Gpp": gpp
    })
    replicate_names.append(sheet)

# Check that all replicates share the same omega values
reference_omega = replicate_data[0]["omega"]
for rep in replicate_data[1:]:
    if len(rep["omega"]) != len(reference_omega) or not np.allclose(rep["omega"], reference_omega, rtol=1e-8, atol=1e-12):
        raise ValueError(
            "All sheets must have the same omega values in the same order "
            "for the mean ± SD plot in this script."
        )

omega = reference_omega

# Stack replicate arrays
Gp_matrix = np.vstack([rep["Gp"] for rep in replicate_data])
Gpp_matrix = np.vstack([rep["Gpp"] for rep in replicate_data])

Gp_mean, Gp_sd = mean_sd(Gp_matrix, axis=0)
Gpp_mean, Gpp_sd = mean_sd(Gpp_matrix, axis=0)


# ============================================================
# FIT EACH REPLICATE
# ============================================================

fit_results = []
for rep in replicate_data:
    fit_out = fit_single_replicate(rep["omega"], rep["Gp"], rep["Gpp"], TAUS)
    fit_results.append({
        "sheet": rep["sheet"],
        **fit_out
    })

param_matrix = np.vstack([fr["params"] for fr in fit_results])   # shape (n_reps, 4)
param_names = ["G_inf", "G1", "G2", "G3"]

param_means, param_cis = mean_ci_95(param_matrix, axis=0)
param_sds = np.std(param_matrix, axis=0, ddof=1) if len(fit_results) > 1 else np.zeros(param_matrix.shape[1])

# Use mean fitted parameters for overlay
gp_fit_meanparams, gpp_fit_meanparams = generalized_maxwell_gp_gpp(omega, param_means, TAUS)

# Residuals corresponding to the displayed overlay curve
Gp_resid_meanplot = compute_relative_residuals(Gp_mean, gp_fit_meanparams)
Gpp_resid_meanplot = compute_relative_residuals(Gpp_mean, gpp_fit_meanparams)

# ============================================================
# SAVE FIT PARAMETERS TO EXCEL
# ============================================================

replicate_param_rows = []
for fr in fit_results:
    row = {
        "replicate": fr["sheet"],
        "G_inf": fr["params"][0],
        "G1": fr["params"][1],
        "tau1": TAUS[0],
        "G2": fr["params"][2],
        "tau2": TAUS[1],
        "G3": fr["params"][3],
        "tau3": TAUS[2],
        "G_total": sum(fr["params"][i] for i in np.arange(4)),
        "cost": fr["cost"],
        "success": fr["success"],
        "nfev": fr["nfev"],
        "message": fr["message"]
    }
    replicate_param_rows.append(row)

replicate_params_df = pd.DataFrame(replicate_param_rows)

summary_df = pd.DataFrame({
    "parameter": ["G_inf", "G1", "tau1", "G2", "tau2", "G3", "tau3"],
    "mean": [
        param_means[0],
        param_means[1], TAUS[0],
        param_means[2], TAUS[1],
        param_means[3], TAUS[2]
    ],
    "sd": [
        param_sds[0],
        param_sds[1], 0.0,
        param_sds[2], 0.0,
        param_sds[3], 0.0
    ],
    "ci95_half_width": [
        param_cis[0],
        param_cis[1], 0.0,
        param_cis[2], 0.0,
        param_cis[3], 0.0
    ]
})

fit_curves_df = pd.DataFrame({
    "omega": omega,
    "Gp_mean_data": Gp_mean,
    "Gp_sd_data": Gp_sd,
    "Gpp_mean_data": Gpp_mean,
    "Gpp_sd_data": Gpp_sd,
    "Gp_fit_from_mean_params": gp_fit_meanparams,
    "Gpp_fit_from_mean_params": gpp_fit_meanparams
})


excel_out = os.path.join(OUTPUT_DIR, f"{strain}_frequency_sweep{time}_gmaxwell3_fit_parameters.xlsx")
with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
    replicate_params_df.to_excel(writer, sheet_name="replicate_fits", index=False)
    summary_df.to_excel(writer, sheet_name="summary", index=False)
    fit_curves_df.to_excel(writer, sheet_name="mean_data_and_fit", index=False)


# ============================================================
# PLOT 1: MEAN MODULI ± SD WITH FIT OVERLAY
# ============================================================

fig, ax = plt.subplots(figsize=(6.0, 5.0))

# Mean data
ax.plot(omega, Gp_mean, marker="o", linestyle="None", label="Mean G'", color=STRAIN_COLORS[strain])
ax.plot(omega, Gpp_mean, marker="s", markerfacecolor='None', linestyle="None", label='Mean G"', color=STRAIN_COLORS[strain])

# SD shading
ax.fill_between(omega, Gp_mean - Gp_sd, Gp_mean + Gp_sd, alpha=ALPHA, color=STRAIN_COLORS[strain])
ax.fill_between(omega, Gpp_mean - Gpp_sd, Gpp_mean + Gpp_sd, alpha=ALPHA, color=STRAIN_COLORS[strain])

# Fit overlay from mean fitted params
ax.plot(omega, gp_fit_meanparams, linestyle="--", linewidth=2, label="Fit", color=FIT_COLOR)
ax.plot(omega, gpp_fit_meanparams, linestyle="--", linewidth=2, color=FIT_COLOR)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Angular frequency ω (rad/s)", fontsize=axis_label_fontsize)
ax.set_ylabel("Shear modulus (MPa)", fontsize=axis_label_fontsize)
ax.set_title(f"{strain}, {time} h: 3-mode fit", fontsize=axis_label_fontsize)

ax.tick_params(axis="both", labelsize=tick_label_fontsize)
ax.legend(fontsize=legend_fontsize, frameon=False)
# ax.grid(True, which="both", alpha=0.3)

fig.tight_layout()

moduli_plot_out = os.path.join(OUTPUT_DIR, f"{strain}_frequency_sweep{time}_gmaxwell3_fit_overlay.svg")
if save_fig:
    fig.savefig(moduli_plot_out, format="svg", bbox_inches="tight")
plt.show()
plt.close(fig)


# ============================================================
# PLOT 2: Gi vs tau_i WITH 95% CI
# ============================================================

# Extract only G1, G2, G3
Gi_means = param_means[1:]
Gi_cis = param_cis[1:]

fig, ax = plt.subplots(figsize=(6.0, 5.0))

ax.errorbar(
    TAUS,
    Gi_means,
    yerr=Gi_cis,
    fmt="o",
    capsize=5,
    linestyle="none"
)

ax.plot(TAUS, Gi_means, linestyle="None", alpha=0.8)

ax.set_xscale("log")
ax.set_ylim(-0.0001,0.00155)
ax.set_xlabel(r"Relaxation time $\tau_i$ (s)", fontsize=axis_label_fontsize)
ax.set_ylabel(r"Mode modulus $G_i$ (MPa)", fontsize=axis_label_fontsize)
ax.set_title(f"{strain}, {time} h: $G_i$ vs $\\tau_i$ (95% CI)", fontsize=axis_label_fontsize)

ax.tick_params(axis="both", labelsize=tick_label_fontsize)
# ax.grid(True, which="both", alpha=0.3)

fig.tight_layout()

gi_plot_out = os.path.join(OUTPUT_DIR, f"{strain}_frequency_sweep{time}_Gi_vs_tau_ci95.svg")
if save_fig:
    fig.savefig(gi_plot_out, format="svg", bbox_inches="tight")
plt.show()
plt.close(fig)

# ============================================================
# PLOT 3: Residual of mean fit to mean data
# ============================================================

fig, ax = plt.subplots(figsize=(6.0, 5.0))

# ax.axhline(0, linestyle="--", linewidth=1)

ax.plot(omega, Gp_resid_meanplot, marker="o", linestyle="-", label="Residual G'", color=STRAIN_COLORS[strain])
ax.plot(omega, Gpp_resid_meanplot, marker="s", markerfacecolor='None', linestyle="-", label='Residual G"', color=STRAIN_COLORS[strain])

ax.set_xscale("log")
ax.set_xlabel("Angular frequency ω (rad/s)", fontsize=axis_label_fontsize)
ax.set_ylim(-.25, .25)
ax.set_ylabel("Relative residual", fontsize=axis_label_fontsize)
ax.set_title(f"{strain}, {time} h: residuals", fontsize=axis_label_fontsize)

ax.tick_params(axis="both", labelsize=tick_label_fontsize)
ax.legend(fontsize=legend_fontsize, frameon=False)
# ax.grid(True, which="both", alpha=0.3)

fig.tight_layout()

resid_overlay_out = os.path.join(
    OUTPUT_DIR,
    f"{strain}_frequency_sweep{time}_gmaxwell3_residuals_mean_overlay.svg"
)
if save_fig:
    fig.savefig(resid_overlay_out, format="svg", bbox_inches="tight")
plt.show()
plt.close(fig)

# ============================================================
# PRINT SUMMARY TO CONSOLE
# ============================================================

print("\nInput file:")
print(input_file)

print("\nReplicate sheets:")
for s in sheet_names:
    print(f"  - {s}")

print("\nMean fitted parameters across replicates:")
for name, val, ci in zip(param_names, param_means, param_cis):
    print(f"{name:>5s} = {val:.6g} ± {ci:.6g} (95% CI half-width)")

print("\nFixed taus:")
for i, taui in enumerate(TAUS, start=1):
    print(f"tau{i} = {taui}")

print("\nSaved files:")
print(excel_out)
print(moduli_plot_out)
# print(gi_plot_out)