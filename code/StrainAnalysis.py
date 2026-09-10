# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 14:19:45 2026

@author: larki
"""

import pandas as pd
import numpy as np
from scipy import stats
# from scipy.stats import mannwhitneyu
# from scipy.stats import sem
# from scipy.stats import t

def cliffs_delta(x, y):
    """
    Nonparametric effect size.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    greater = sum(i > j for i in x for j in y)
    less = sum(i < j for i in x for j in y)

    return (greater - less) / (len(x) * len(y))


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
    ci_low = mean - ci
    ci_high = mean + ci
    return float(ci_low), float(ci_high)


def ci_nonoverlap(ci1_low, ci1_high, ci2_low, ci2_high):
    """
    Returns True if confidence intervals do NOT overlap.
    """
    return (ci1_high < ci2_low) or (ci2_high < ci1_low)


def safe_welch_p(x1, x2):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]

    if len(x1) < 2 or len(x2) < 2:
        return np.nan
    return stats.ttest_ind(x1, x2, equal_var=False).pvalue


def compare_mutants_wide(
    file_A,
    file_B,
    label_A="Mutant_A",
    label_B="Mutant_B",
    alpha=0.05
):
    """
    Compare replicate-level rheological quantities
    from two mutants (wide format).
    """
    dfA = pd.read_excel(file_A)
    dfB = pd.read_excel(file_B)

    # Identify quantity columns (exclude replicate ID)
    exclude = {'replicate'}
    quantities = sorted(
        set(dfA.columns).intersection(dfB.columns) - exclude
    )
    
    results = []

    for q in quantities:
        valsA = dfA[q].dropna().values
        valsB = dfB[q].dropna().values
        
        ci_lowA, ci_highA = mean_ci(valsA)
        ci_lowB, ci_highB = mean_ci(valsB)
        
        not_overlap = ci_nonoverlap(ci_lowA, ci_highA, ci_lowB, ci_lowB)

        if len(valsA) < 2 or len(valsB) < 2:
            continue

        # Mann–Whitney U test
        # p = bootstrap_hypothesis_test(valsA, valsB, n_boot=100000)
        p = safe_welch_p(valsA, valsB)

        delta = cliffs_delta(valsA, valsB)

        meanA, meanB = np.mean(valsA), np.mean(valsB)

        results.append({
            "quantity": q,
            f"{label_A}_mean": meanA,
            f"{label_A}_sd": np.std(valsA, ddof=1),
            f"{label_B}_mean": meanB,
            f"{label_B}_sd": np.std(valsB, ddof=1),
            "mean_difference": meanB - meanA,
            "fold_change": meanB / meanA if meanA != 0 else np.nan,
            "p_value": p,
            "statistically_significant": p < alpha,
            "cliffs_delta": delta,
            f"{label_A}_ci_low": ci_lowA,
            f"{label_A}_ci_high": ci_highA,
            f"{label_B}_ci_low": ci_lowB,
            f"{label_B}_ci_high": ci_highB,
            "CI non-overlap": not_overlap
        })

    res_df = pd.DataFrame(results).set_index("quantity")

    # Sort: significant + large effects first
    res_df = res_df.sort_values(
        by=["statistically_significant", "cliffs_delta"],
        ascending=[False, False]
    )

    return res_df

path = 'H:/Analysis/In process/Strain sweep'

file_name1 = '1233_24hpi_analysis_summary'
file_name2 = '1480_24hpi_analysis_summary'

file_path1 = f'{path}/{file_name1}.xlsx'
file_path2 = f'{path}/{file_name2}.xlsx'

strain_dictionary = dict([('1233','WT'),('1234','ΔPs'),('1250','ΔpgsB'),('1480','ΔepsA-O')])

label1 = strain_dictionary[file_name1[:4]]
label2 = strain_dictionary[file_name2[:4]]

comparison = compare_mutants_wide(
    file_path1,
    file_path2,
    label_A=label1,
    label_B=label2
)

print(comparison)
comparison.to_excel(f"{path}/{label1}_vs_{label2}/{label1}_vs_{label2}_rheology_stats.xlsx")
