"""Classical A/B-test analysis of a randomised experiment."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def srm_test(t, expected_share: float) -> dict:
    """Sample-ratio mismatch: is the observed treatment share consistent with the design?"""
    t = np.asarray(t)
    n1, n = int(t.sum()), len(t)
    chi2, p = stats.chisquare([n1, n - n1], [n * expected_share, n * (1 - expected_share)])
    return {"n": n, "n_treated": n1, "share_treated": n1 / n, "expected": expected_share, "chi2": float(chi2), "p_value": float(p)}


def balance_table(df: pd.DataFrame, features, t_col="treatment") -> pd.DataFrame:
    """Standardised mean differences (|SMD| < 0.1 is the usual 'balanced' threshold)."""
    rows = []
    g1, g0 = df[df[t_col] == 1], df[df[t_col] == 0]
    for f in features:
        m1, m0 = g1[f].mean(), g0[f].mean()
        sd = np.sqrt((g1[f].var() + g0[f].var()) / 2)
        rows.append({"feature": f, "mean_treated": m1, "mean_control": m0, "smd": (m1 - m0) / sd if sd > 0 else 0.0})
    return pd.DataFrame(rows).set_index("feature")


def ate_binary(y, t, alpha: float = 0.05) -> dict:
    """Difference in proportions with a Wald CI, relative lift, and a two-proportion z-test."""
    y, t = np.asarray(y), np.asarray(t)
    y1, y0 = y[t == 1], y[t == 0]
    p1, p0 = y1.mean(), y0.mean()
    se = np.sqrt(p1 * (1 - p1) / len(y1) + p0 * (1 - p0) / len(y0))
    z = stats.norm.ppf(1 - alpha / 2)
    pooled = y.mean()
    se0 = np.sqrt(pooled * (1 - pooled) * (1 / len(y1) + 1 / len(y0)))
    zstat = (p1 - p0) / se0 if se0 > 0 else np.nan
    return {"rate_treated": p1, "rate_control": p0, "ate": p1 - p0, "ci_low": p1 - p0 - z * se,
            "ci_high": p1 - p0 + z * se, "relative_lift": (p1 - p0) / p0 if p0 > 0 else np.nan,
            "z": zstat, "p_value": 2 * (1 - stats.norm.cdf(abs(zstat)))}


def mde(p0: float, n_treated: int, n_control: int, alpha=0.05, power=0.8) -> float:
    """Minimum detectable absolute effect for a two-proportion test."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * np.sqrt(p0 * (1 - p0) * (1 / n_treated + 1 / n_control)))


def required_n_per_group(p0: float, abs_effect: float, alpha=0.05, power=0.8, ratio: float = 1.0) -> int:
    """Control-group size needed (treated = ratio * control) to detect `abs_effect`."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    p1 = p0 + abs_effect
    var = p0 * (1 - p0) + p1 * (1 - p1) / ratio
    return int(np.ceil(z ** 2 * var / abs_effect ** 2))


def effect_by_quantile(df: pd.DataFrame, feature: str, y_col: str, q: int = 5, t_col="treatment") -> pd.DataFrame:
    """Subgroup ATEs by value-based quantile bins of one feature (tied values stay in the same bin; heavily tied
    features give fewer bins). Bins lacking either arm are skipped."""
    bins = pd.qcut(df[feature], q, duplicates="drop")
    rows = []
    for i, (b, d) in enumerate(df.groupby(bins, observed=True)):
        if d[t_col].nunique() < 2:
            continue
        r = ate_binary(d[y_col], d[t_col])
        rows.append({"bin": f"B{i+1}", "range": str(b), "n": len(d),
                     **{k: r[k] for k in ["rate_control", "ate", "ci_low", "ci_high"]}})
    return pd.DataFrame(rows).set_index("bin") if rows else pd.DataFrame(columns=["ate", "ci_low", "ci_high"])
