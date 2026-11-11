"""Uplift evaluation on randomised data: uplift/Qini curves, AUUC, Qini coefficient, uplift@k, bootstrap CIs,
and a budget-aware targeting policy."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _cum(y, t, score):
    o = np.argsort(-np.asarray(score), kind="mergesort")
    y, t = np.asarray(y, dtype=float)[o], np.asarray(t, dtype=float)[o]
    nt, nc = np.cumsum(t), np.cumsum(1 - t)
    yt, yc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    return nt, nc, yt, yc


def qini_curve(y, t, score):
    """Q(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k): incremental responders among the top-k (in units of the treated
    group). Returns x = fraction targeted (0..1), q."""
    nt, nc, yt, yc = _cum(y, t, score)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = yt - np.where(nc > 0, yc * nt / nc, 0.0)
    x = np.arange(1, len(q) + 1) / len(q)
    return np.r_[0, x], np.r_[0, q]


def uplift_curve(y, t, score):
    """U(k) = (Y_t(k)/N_t(k) - Y_c(k)/N_c(k)) * k : incremental responders if the top-k were treated."""
    nt, nc, yt, yc = _cum(y, t, score)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (np.where(nt > 0, yt / nt, 0) - np.where(nc > 0, yc / nc, 0)) * np.arange(1, len(nt) + 1)
    x = np.arange(1, len(u) + 1) / len(u)
    return np.r_[0, x], np.r_[0, u]


def _area(x, yv):
    return float(np.trapezoid(yv, x)) if hasattr(np, "trapezoid") else float(np.trapz(yv, x))


def qini_coefficient(y, t, score) -> float:
    """(area under Qini curve - random) / (area under the perfect Qini curve - random). 0 = random, 1 = perfect."""
    y, t = np.asarray(y), np.asarray(t)
    x, q = qini_curve(y, t, score)
    rand = _area(x, x * q[-1])
    perfect_score = y * t - y * (1 - t)  # treated responders first, control responders last
    xp, qp = qini_curve(y, t, perfect_score)
    return (_area(x, q) - rand) / (_area(xp, qp) - rand)


def auuc(y, t, score) -> float:
    """Area between the uplift curve and the random line, per user (higher = better)."""
    x, u = uplift_curve(y, t, score)
    return (_area(x, u) - _area(x, x * u[-1])) / len(y)


def uplift_at_k(y, t, score, k: float = 0.3) -> float:
    """Treated-minus-control response rate among the top-k fraction ranked by score."""
    o = np.argsort(-np.asarray(score), kind="mergesort")[: int(len(y) * k)]
    y, t = np.asarray(y)[o], np.asarray(t)[o]
    return float(y[t == 1].mean() - y[t == 0].mean())


def summary(y, t, score, k_list=(0.1, 0.3)) -> dict:
    out = {"qini_coef": qini_coefficient(y, t, score), "auuc": auuc(y, t, score)}
    for k in k_list:
        out[f"uplift@{int(k*100)}%"] = uplift_at_k(y, t, score, k)
    return out


def bootstrap(y, t, score, fn=qini_coefficient, n_boot: int = 200, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    y, t, score = map(np.asarray, (y, t, score))
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        vals.append(fn(y[i], t[i], score[i]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(vals)), float(lo), float(hi)


def decile_table(y, t, score, n_bins: int = 10) -> pd.DataFrame:
    """Observed treated-minus-control rate per predicted-uplift decile (calibration of the ranking)."""
    df = pd.DataFrame({"y": y, "t": t, "s": score})
    df["decile"] = pd.qcut(df.s.rank(method="first", ascending=False), n_bins, labels=range(1, n_bins + 1))
    g = df.groupby("decile", observed=True)
    return pd.DataFrame({
        "mean_pred": g.s.mean(),
        "rate_treated": g.apply(lambda d: d.y[d.t == 1].mean()),
        "rate_control": g.apply(lambda d: d.y[d.t == 0].mean()),
    }).assign(observed_uplift=lambda d: d.rate_treated - d.rate_control)


def policy_curve(y, t, score, value_per_response: float, cost_per_treated: float, grid=None) -> pd.DataFrame:
    """Expected profit per 1,000 users if we treat the top-k fraction (incremental value - targeting cost).
    Incremental responses are estimated from the randomised data within the top-k."""
    y, t, score = map(np.asarray, (y, t, score))
    grid = np.linspace(0.05, 1.0, 20) if grid is None else grid
    o = np.argsort(-score, kind="mergesort")
    rows = []
    for k in grid:
        top = o[: max(1, int(len(y) * k))]
        yt, tt = y[top], t[top]
        up = yt[tt == 1].mean() - yt[tt == 0].mean()
        rows.append({"target_frac": k, "uplift_in_segment": up,
                     "profit_per_1000": 1000 * k * (value_per_response * up - cost_per_treated)})
    return pd.DataFrame(rows)
