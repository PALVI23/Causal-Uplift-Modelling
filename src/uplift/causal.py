"""Two experiments that need ground truth:
1. Semi-synthetic benchmark - real covariates, simulated outcomes with a KNOWN individual effect tau(x),
   so learners can be scored on PEHE (error in the individual effect) and not only on ranking.
2. Observational-bias experiment - turn the RCT into confounded observational data by selective sub-sampling
   that keeps the covariate distribution unchanged, then compare naive / regression / IPW / AIPW estimates of
   the ATE against the randomised benchmark."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold


# ------------------------------------------------------------------------------ 1. semi-synthetic
def semi_synthetic(X: np.ndarray, base_logit: np.ndarray, scenario: str = "heterogeneous", treat_share=0.85,
                   seed: int = 0):
    """Potential outcomes on the probability scale with a known effect.
    base_logit: realistic baseline log-odds (e.g. from a model fit on real control users).
    scenarios: 'heterogeneous' (effect varies strongly, negative for some users), 'weak' (small, nearly constant)."""
    rng = np.random.default_rng(seed)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    if scenario == "heterogeneous":
        delta = 0.9 * np.tanh(Z[:, 2]) + 0.6 * (Z[:, 8] > 0.5) - 0.7 * (Z[:, 4] > 1.0) + 0.3 * Z[:, 0] * (Z[:, 1] > 0)
    elif scenario == "weak":
        delta = 0.25 + 0.1 * np.tanh(Z[:, 2])
    else:
        raise ValueError(scenario)
    p0 = expit(base_logit)
    p1 = expit(base_logit + delta)
    t = (rng.random(len(X)) < treat_share).astype(int)
    y = (rng.random(len(X)) < np.where(t == 1, p1, p0)).astype(int)
    return t, y, p1 - p0


def pehe(tau_hat, tau_true) -> float:
    """Precision in Estimating Heterogeneous Effects (Hill, 2011): RMSE of individual effects."""
    return float(np.sqrt(np.mean((np.asarray(tau_hat) - np.asarray(tau_true)) ** 2)))


# ------------------------------------------------------------------------------ 2. observational bias
def confounded_subsample(t: np.ndarray, propensity: np.ndarray, treat_share: float, seed: int = 0) -> np.ndarray:
    """Keep each unit with probability depending on (t, x) such that in the kept sample
    P(T=1 | x) = propensity(x) while the covariate distribution is UNCHANGED (so the target ATE is the
    same as in the full randomised data). Requires propensity in (0, 1)."""
    rng = np.random.default_rng(seed)
    pt, pc = treat_share, 1 - treat_share
    s = np.asarray(propensity)
    c = min(1.0 / pt, pc / pt / (1 - s.min())) * 0.999   # largest scale keeping both keep-probabilities <= 1
    keep_t = c * s
    keep_c = c * (1 - s) * pt / pc
    keep = np.where(t == 1, keep_t, keep_c)
    return rng.random(len(t)) < keep


def _crossfit_nuisances(X, t, y, folds=2, seed=0):
    mu0, mu1, e = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
    p = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200, verbose=-1, random_state=seed)
    for a, b in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, t * 2 + y):
        m1 = lgb.LGBMClassifier(**p).fit(X[a][t[a] == 1], y[a][t[a] == 1])
        m0 = lgb.LGBMClassifier(**p).fit(X[a][t[a] == 0], y[a][t[a] == 0])
        g = lgb.LGBMClassifier(**p).fit(X[a], t[a])
        mu1[b], mu0[b], e[b] = m1.predict_proba(X[b])[:, 1], m0.predict_proba(X[b])[:, 1], g.predict_proba(X[b])[:, 1]
    return mu0, mu1, np.clip(e, 0.01, 0.99)


def ate_estimators(X, t, y, oracle_propensity=None, seed=0, alpha=0.05) -> pd.DataFrame:
    """Naive difference in means, outcome regression (T-learner average), IPW (Hajek), and AIPW (doubly robust,
    cross-fitted) with influence-function / delta-method CIs."""
    X, t, y = np.asarray(X), np.asarray(t), np.asarray(y).astype(float)
    z = stats.norm.ppf(1 - alpha / 2)
    rows = []
    y1, y0 = y[t == 1], y[t == 0]
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    rows.append(("Naive difference in means", y1.mean() - y0.mean(), se))
    mu0, mu1, e_hat = _crossfit_nuisances(X, t, y, seed=seed)
    reg = mu1 - mu0
    rows.append(("Outcome regression", reg.mean(), reg.std(ddof=1) / np.sqrt(len(y))))

    def hajek(e):
        w1, w0 = t / e, (1 - t) / (1 - e)
        est = (w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum()
        infl = w1 * (y - (w1 * y).sum() / w1.sum()) / w1.mean() - w0 * (y - (w0 * y).sum() / w0.sum()) / w0.mean()
        return est, infl.std(ddof=1) / np.sqrt(len(y))

    rows.append(("IPW (estimated propensity)", *hajek(e_hat)))
    if oracle_propensity is not None:
        rows.append(("IPW (true propensity)", *hajek(np.asarray(oracle_propensity))))
    phi = mu1 - mu0 + t * (y - mu1) / e_hat - (1 - t) * (y - mu0) / (1 - e_hat)
    rows.append(("AIPW / doubly robust", phi.mean(), phi.std(ddof=1) / np.sqrt(len(y))))
    df = pd.DataFrame(rows, columns=["estimator", "ate", "se"])
    df["ci_low"], df["ci_high"] = df.ate - z * df.se, df.ate + z * df.se
    return df.set_index("estimator")


def baseline_logit_model(X, y, seed=0):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200, verbose=-1,
                           random_state=seed).fit(X, y)
    return lambda Xn: logit(np.clip(m.predict_proba(Xn)[:, 1], 1e-5, 1 - 1e-5))
