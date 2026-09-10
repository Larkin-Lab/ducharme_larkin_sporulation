# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:02:52 2026

@author: LarkinLab
"""

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
input_file = os.path.join(
    DATA_DIR,
    f"{strain}_frequency_sweep{time}.xlsx"
)

# Output folder
OUTPUT_DIR = os.path.join(DATA_DIR, "fit_g")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# MODEL SETTINGS
# ============================================================

# Two separate models will be fit:
#
# Single-mode:
#     G_inf + G1, tau1 = 0.01 s
#
# Double-mode:
#     G_inf + G1 + G2
#     tau1 = 0.01 s
#     tau2 = 0.1 s

MODELS = {
    "single": {
        "taus": np.array([0.01], dtype=float),
        "label": "1-mode"
    },

    "double": {
        "taus": np.array([0.01, 0.1], dtype=float),
        "label": "2-mode"
    }
}


# Optional plot settings
axis_label_fontsize = 20
tick_label_fontsize = 18
legend_fontsize = 14
ALPHA = 0.1


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
    Generalized Maxwell model with fixed taus and fitted parameters.

    params = [G_inf, G1, G2, ...]
    
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

    gp_fit, gpp_fit = generalized_maxwell_gp_gpp(
        omega,
        params,
        taus
    )

    if mode == "relative":

        eps = 1e-15

        r_gp = (
            (gp_fit - gp_obs)
            / np.maximum(np.abs(gp_obs), eps)
        )

        r_gpp = (
            (gpp_fit - gpp_obs)
            / np.maximum(np.abs(gpp_obs), eps)
        )

    else:

        r_gp = gp_fit - gp_obs
        r_gpp = gpp_fit - gpp_obs

    return np.concatenate([r_gp, r_gpp])


def initial_guess_from_data(omega, gp, gpp, n_modes):
    """
    Heuristic initial guess for:

        [G_inf, G1, G2, ...]

    The available modulus span is divided among the
    finite relaxation modes.
    """

    gp = np.asarray(gp, dtype=float)
    gpp = np.asarray(gpp, dtype=float)

    G_inf0 = max(np.min(gp) * 0.5, 0.0)

    span = max(
        np.max(gp) - G_inf0,
        np.max(gpp),
        1e-10
    )

    if n_modes == 1:

        G1_0 = span

        return np.array(
            [G_inf0, G1_0],
            dtype=float
        )

    elif n_modes == 2:

        G1_0 = 0.6 * span
        G2_0 = 0.4 * span

        return np.array(
            [G_inf0, G1_0, G2_0],
            dtype=float
        )

    else:

        raise ValueError(
            "Only single- and double-mode fits are supported."
        )


def fit_single_replicate(omega, gp, gpp, taus):
    """
    Fit one replicate and return fitted params and fitted curves.
    """

    n_modes = len(taus)

    x0 = initial_guess_from_data(
        omega,
        gp,
        gpp,
        n_modes
    )

    # One parameter for G_inf plus one for each finite mode
    n_params = 1 + n_modes

    LOWER_BOUNDS = np.zeros(n_params, dtype=float)

    UPPER_BOUNDS = np.full(
        n_params,
        np.inf,
        dtype=float
    )

    result = least_squares(
        residuals,
        x0=x0,
        bounds=(
            LOWER_BOUNDS,
            UPPER_BOUNDS
        ),
        args=(
            omega,
            gp,
            gpp,
            taus,
            "relative"
        ),
        method="trf",
        max_nfev=20000
    )

    params = result.x

    gp_fit, gpp_fit = generalized_maxwell_gp_gpp(
        omega,
        params,
        taus
    )

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

    sd = (
        np.std(arr, axis=axis, ddof=1)
        if arr.shape[axis] > 1
        else np.zeros_like(mean)
    )

    return mean, sd


def mean_ci_95(arr, axis=0):

    arr = np.asarray(arr, dtype=float)

    n = arr.shape[axis]

    mean = np.mean(arr, axis=axis)

    if n <= 1:

        ci = np.zeros_like(mean)

        return mean, ci

    sd = np.std(
        arr,
        axis=axis,
        ddof=1
    )

    sem = sd / np.sqrt(n)

    tcrit = t.ppf(
        0.975,
        df=n - 1
    )

    ci = tcrit * sem

    return mean, ci


def compute_relative_residuals(g_obs, g_fit):

    eps = 1e-15

    return (
        (g_fit - g_obs)
        / np.maximum(np.abs(g_obs), eps)
    )


# ============================================================
# READ DATA
# ============================================================

if not os.path.exists(input_file):

    raise FileNotFoundError(
        f"Could not find input file:\n{input_file}"
    )


excel_file = pd.ExcelFile(input_file)

sheet_names = excel_file.sheet_names

if len(sheet_names) == 0:

    raise ValueError(
        "No sheets found in workbook."
    )


replicate_data = []
replicate_names = []


for sheet in sheet_names:

    df = pd.read_excel(
        input_file,
        sheet_name=sheet
    )

    expected_cols = {
        "omega",
        "Gp",
        "Gpp"
    }

    if not expected_cols.issubset(df.columns):

        raise ValueError(
            f"Sheet '{sheet}' must contain columns "
            f"{expected_cols}, but has columns: "
            f"{list(df.columns)}"
        )

    df = df[
        ["omega", "Gp", "Gpp"]
    ].dropna().copy()

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


# ============================================================
# CHECK OMEGA VALUES
# ============================================================

reference_omega = replicate_data[0]["omega"]

for rep in replicate_data[1:]:

    if (
        len(rep["omega"]) != len(reference_omega)
        or not np.allclose(
            rep["omega"],
            reference_omega,
            rtol=1e-8,
            atol=1e-12
        )
    ):

        raise ValueError(
            "All sheets must have the same omega values "
            "in the same order for the mean ± SD plot "
            "in this script."
        )


omega = reference_omega


# ============================================================
# STACK REPLICATE ARRAYS
# ============================================================

Gp_matrix = np.vstack(
    [rep["Gp"] for rep in replicate_data]
)

Gpp_matrix = np.vstack(
    [rep["Gpp"] for rep in replicate_data]
)


Gp_mean, Gp_sd = mean_sd(
    Gp_matrix,
    axis=0
)

Gpp_mean, Gpp_sd = mean_sd(
    Gpp_matrix,
    axis=0
)


# ============================================================
# RUN BOTH MODELS
# ============================================================

for model_name, model_settings in MODELS.items():

    TAUS = model_settings["taus"]
    model_label = model_settings["label"]

    n_modes = len(TAUS)

    print("\n")
    print("=" * 70)
    print(f"RUNNING {model_label.upper()} GENERALIZED MAXWELL FIT")
    print("=" * 70)

    # ========================================================
    # FIT EACH REPLICATE
    # ========================================================

    fit_results = []

    for rep in replicate_data:

        fit_out = fit_single_replicate(
            rep["omega"],
            rep["Gp"],
            rep["Gpp"],
            TAUS
        )

        fit_results.append({
            "sheet": rep["sheet"],
            **fit_out
        })


    # ========================================================
    # FIT PARAMETERS
    # ========================================================

    param_matrix = np.vstack(
        [
            fr["params"]
            for fr in fit_results
        ]
    )


    # Parameter names:
    # [G_inf, G1] for single mode
    # [G_inf, G1, G2] for double mode

    param_names = ["G_inf"]

    for i in range(n_modes):
        param_names.append(f"G{i + 1}")


    param_means, param_cis = mean_ci_95(
        param_matrix,
        axis=0
    )


    param_sds = (
        np.std(
            param_matrix,
            axis=0,
            ddof=1
        )
        if len(fit_results) > 1
        else np.zeros(
            param_matrix.shape[1]
        )
    )


    # ========================================================
    # MEAN FITTED PARAMETERS
    # ========================================================

    gp_fit_meanparams, gpp_fit_meanparams = (
        generalized_maxwell_gp_gpp(
            omega,
            param_means,
            TAUS
        )
    )


    # ========================================================
    # RESIDUALS
    # ========================================================

    Gp_resid_meanplot = compute_relative_residuals(
        Gp_mean,
        gp_fit_meanparams
    )

    Gpp_resid_meanplot = compute_relative_residuals(
        Gpp_mean,
        gpp_fit_meanparams
    )


    # ========================================================
    # SAVE FIT PARAMETERS TO EXCEL
    # ========================================================

    replicate_param_rows = []


    for fr in fit_results:

        params = fr["params"]

        row = {
            "replicate": fr["sheet"],
            "G_inf": params[0]
        }


        # Add finite mode parameters
        for i, taui in enumerate(TAUS):

            row[f"G{i + 1}"] = params[i + 1]

            row[f"tau{i + 1}"] = taui


        # G_total = G_inf + all finite modes
        row["G_total"] = np.sum(params)

        row["cost"] = fr["cost"]

        row["success"] = fr["success"]

        row["nfev"] = fr["nfev"]

        row["message"] = fr["message"]

        replicate_param_rows.append(row)


    replicate_params_df = pd.DataFrame(
        replicate_param_rows
    )


    # ========================================================
    # SUMMARY TABLE
    # ========================================================

    summary_rows = []

    # G_inf
    summary_rows.append({
        "parameter": "G_inf",
        "mean": param_means[0],
        "sd": param_sds[0],
        "ci95_half_width": param_cis[0]
    })


    # Finite modes
    for i, taui in enumerate(TAUS):

        summary_rows.append({
            "parameter": f"G{i + 1}",
            "mean": param_means[i + 1],
            "sd": param_sds[i + 1],
            "ci95_half_width": param_cis[i + 1]
        })

        summary_rows.append({
            "parameter": f"tau{i + 1}",
            "mean": taui,
            "sd": 0.0,
            "ci95_half_width": 0.0
        })


    summary_df = pd.DataFrame(
        summary_rows
    )


    # ========================================================
    # FIT CURVES TABLE
    # ========================================================

    fit_curves_df = pd.DataFrame({

        "omega": omega,

        "Gp_mean_data": Gp_mean,

        "Gp_sd_data": Gp_sd,

        "Gpp_mean_data": Gpp_mean,

        "Gpp_sd_data": Gpp_sd,

        "Gp_fit_from_mean_params":
            gp_fit_meanparams,

        "Gpp_fit_from_mean_params":
            gpp_fit_meanparams
    })


    # ========================================================
    # SAVE EXCEL FILE
    # ========================================================

    excel_out = os.path.join(
        OUTPUT_DIR,
        f"{strain}_frequency_sweep{time}_"
        f"gmaxwell{n_modes}_fit_parameters.xlsx"
    )


    with pd.ExcelWriter(
        excel_out,
        engine="openpyxl"
    ) as writer:

        replicate_params_df.to_excel(
            writer,
            sheet_name="replicate_fits",
            index=False
        )

        summary_df.to_excel(
            writer,
            sheet_name="summary",
            index=False
        )

        fit_curves_df.to_excel(
            writer,
            sheet_name="mean_data_and_fit",
            index=False
        )


    # ========================================================
    # PLOT 1:
    # MEAN MODULI ± SD WITH FIT OVERLAY
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6.0, 5.0)
    )


    # Mean data
    ax.plot(
        omega,
        Gp_mean,
        marker="o",
        linestyle="None",
        label="Mean G'",
        color=STRAIN_COLORS[strain]
    )


    ax.plot(
        omega,
        Gpp_mean,
        marker="s",
        markerfacecolor="None",
        linestyle="None",
        label='Mean G"',
        color=STRAIN_COLORS[strain]
    )


    # SD shading
    ax.fill_between(
        omega,
        Gp_mean - Gp_sd,
        Gp_mean + Gp_sd,
        alpha=ALPHA,
        color=STRAIN_COLORS[strain]
    )


    ax.fill_between(
        omega,
        Gpp_mean - Gpp_sd,
        Gpp_mean + Gpp_sd,
        alpha=ALPHA,
        color=STRAIN_COLORS[strain]
    )


    # Fit overlay
    ax.plot(
        omega,
        gp_fit_meanparams,
        linestyle="--",
        linewidth=2,
        label="Fit",
        color=FIT_COLOR
    )


    ax.plot(
        omega,
        gpp_fit_meanparams,
        linestyle="--",
        linewidth=2,
        color=FIT_COLOR
    )


    ax.set_xscale("log")
    ax.set_yscale("log")


    ax.set_xlabel(
        "Angular frequency ω (rad/s)",
        fontsize=axis_label_fontsize
    )


    ax.set_ylabel(
        "Shear modulus (MPa)",
        fontsize=axis_label_fontsize
    )


    ax.set_title(
        f"{strain}, {time} h: "
        f"{model_label} fit",
        fontsize=axis_label_fontsize
    )


    ax.tick_params(
        axis="both",
        labelsize=tick_label_fontsize
    )


    ax.legend(
        fontsize=legend_fontsize,
        frameon=False
    )


    fig.tight_layout()


    moduli_plot_out = os.path.join(
        OUTPUT_DIR,
        f"{strain}_frequency_sweep{time}_"
        f"gmaxwell{n_modes}_fit_overlay.svg"
    )


    if save_fig:

        fig.savefig(
            moduli_plot_out,
            format="svg",
            bbox_inches="tight"
        )


    plt.show()
    plt.close(fig)


    # ========================================================
    # PLOT 2:
    # G_inf AND Gi vs tau_i WITH 95% CI
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6.0, 5.0)
    )


    # --------------------------------------------------------
    # Finite modes
    # --------------------------------------------------------

    finite_G_means = param_means[1:]
    finite_G_cis = param_cis[1:]


    ax.errorbar(
        TAUS,
        finite_G_means,
        yerr=finite_G_cis,
        fmt="o",
        capsize=5,
        linestyle="none",
        label="Finite modes"
    )


    # --------------------------------------------------------
    # G_inf
    #
    # Since G_inf corresponds to tau -> infinity, place it
    # at a pseudo-position to the right of the finite modes.
    # --------------------------------------------------------

    if n_modes == 1:

        GINF_X = 0.1

    else:

        GINF_X = 0.2


    ax.errorbar(
        GINF_X,
        param_means[0],
        yerr=param_cis[0],
        fmt="s",
        capsize=5,
        linestyle="none",
        label=r"$G_\infty$"
    )


    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xscale("log")


    ax.set_xlabel(
        r"Relaxation time $\tau_i$ (s)",
        fontsize=axis_label_fontsize
    )


    ax.set_ylabel(
        r"Mode modulus $G_i$ (MPa)",
        fontsize=axis_label_fontsize
    )


    ax.set_title(
        f"{strain}, {time} h: "
        f"{model_label} $G_i$ vs $\\tau_i$ (95% CI)",
        fontsize=axis_label_fontsize
    )


    ax.tick_params(
        axis="both",
        labelsize=tick_label_fontsize
    )


    ax.legend(
        fontsize=legend_fontsize,
        frameon=False
    )


    fig.tight_layout()


    gi_plot_out = os.path.join(
        OUTPUT_DIR,
        f"{strain}_frequency_sweep{time}_"
        f"gmaxwell{n_modes}_Gi_vs_tau_ci95.svg"
    )


    if save_fig:

        fig.savefig(
            gi_plot_out,
            format="svg",
            bbox_inches="tight"
        )


    plt.show()
    plt.close(fig)


    # ========================================================
    # PLOT 3:
    # RESIDUAL OF MEAN FIT TO MEAN DATA
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6.0, 5.0)
    )


    ax.plot(
        omega,
        Gp_resid_meanplot,
        marker="o",
        linestyle="-",
        label="Residual G'",
        color=STRAIN_COLORS[strain]
    )


    ax.plot(
        omega,
        Gpp_resid_meanplot,
        marker="s",
        markerfacecolor="None",
        linestyle="-",
        label='Residual G"',
        color=STRAIN_COLORS[strain]
    )


    ax.set_xscale("log")


    ax.set_xlabel(
        "Angular frequency ω (rad/s)",
        fontsize=axis_label_fontsize
    )


    ax.set_ylim(
        -1.0,
        1.0
    )


    ax.set_ylabel(
        "Relative residual",
        fontsize=axis_label_fontsize
    )


    ax.set_title(
        f"{strain}, {time} h: "
        f"residuals of {model_label}",
        fontsize=axis_label_fontsize
    )


    ax.tick_params(
        axis="both",
        labelsize=tick_label_fontsize
    )


    ax.legend(
        fontsize=legend_fontsize,
        frameon=False
    )


    fig.tight_layout()


    resid_overlay_out = os.path.join(
        OUTPUT_DIR,
        f"{strain}_frequency_sweep{time}_"
        f"gmaxwell{n_modes}_residuals_mean_overlay.svg"
    )


    if save_fig:

        fig.savefig(
            resid_overlay_out,
            format="svg",
            bbox_inches="tight"
        )


    plt.show()
    plt.close(fig)


    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\nInput file:")
    print(input_file)


    print("\nModel:")
    print(model_label)


    print("\nReplicate sheets:")

    for s in sheet_names:
        print(f"  - {s}")


    print("\nMean fitted parameters across replicates:")

    for name, val, ci in zip(
        param_names,
        param_means,
        param_cis
    ):

        print(
            f"{name:>5s} = "
            f"{val:.6g} ± {ci:.6g} "
            f"(95% CI half-width)"
        )


    print("\nFixed taus:")

    for i, taui in enumerate(
        TAUS,
        start=1
    ):

        print(
            f"tau{i} = {taui}"
        )


    print("\nSaved files:")

    print(excel_out)
    print(moduli_plot_out)
    print(gi_plot_out)
    print(resid_overlay_out)