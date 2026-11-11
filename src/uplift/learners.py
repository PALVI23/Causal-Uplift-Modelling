"""Meta-learners for the conditional average treatment effect (CATE) tau(x) = E[Y(1) - Y(0) | X = x].
All share the API: fit(X, t, y) -> self; predict(X) -> tau_hat. LightGBM base learners."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold

CLF = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=-1)
REG = dict(CLF, min_child_samples=1000, num_leaves=15, n_estimators=200)  # pseudo-outcomes are noisy


def _clf(seed):
    return lgb.LGBMClassifier(**CLF, random_state=seed)


def _reg(seed):
    return lgb.LGBMRegressor(**REG, random_state=seed)


class SLearner:
    """One model mu(x, t); tau = mu(x,1) - mu(x,0). Tends to shrink effects towards 0 (t is 'just a feature')."""
    name = "S-learner"

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, t, y):
        self.m = _clf(self.seed).fit(np.column_stack([X, t]), y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        p1 = self.m.predict_proba(np.column_stack([X, np.ones(len(X))]))[:, 1]
        p0 = self.m.predict_proba(np.column_stack([X, np.zeros(len(X))]))[:, 1]
        return p1 - p0


class TLearner:
    """Separate outcome models per arm; tau = mu1(x) - mu0(x). Unbiased but noisy for small arms."""
    name = "T-learner"

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, t, y):
        X, t, y = np.asarray(X), np.asarray(t), np.asarray(y)
        self.m1 = _clf(self.seed).fit(X[t == 1], y[t == 1])
        self.m0 = _clf(self.seed).fit(X[t == 0], y[t == 0])
        return self

    def predict(self, X):
        return self.m1.predict_proba(X)[:, 1] - self.m0.predict_proba(X)[:, 1]


class XLearner:
    """Kuenzel et al. (2019): impute individual effects with the other arm's model, regress them, and
    combine with the propensity. Designed for unbalanced arms (here 85/15)."""
    name = "X-learner"

    def __init__(self, seed=0, propensity: float | None = None):
        self.seed, self.e = seed, propensity

    def fit(self, X, t, y):
        X, t, y = np.asarray(X), np.asarray(t), np.asarray(y)
        self.e = self.e if self.e is not None else t.mean()
        m1 = _clf(self.seed).fit(X[t == 1], y[t == 1])
        m0 = _clf(self.seed).fit(X[t == 0], y[t == 0])
        d1 = y[t == 1] - m0.predict_proba(X[t == 1])[:, 1]
        d0 = m1.predict_proba(X[t == 0])[:, 1] - y[t == 0]
        self.g1 = _reg(self.seed).fit(X[t == 1], d1)
        self.g0 = _reg(self.seed).fit(X[t == 0], d0)
        return self

    def predict(self, X):
        # weight the estimate built from the larger arm's imputation more: tau = e*tau0 + (1-e)*tau1
        return self.e * self.g0.predict(X) + (1 - self.e) * self.g1.predict(X)


class DRLearner:
    """Kennedy (2020) doubly-robust learner with 2-fold cross-fitting: regress the AIPW pseudo-outcome
    phi = mu1 - mu0 + t(y - mu1)/e - (1-t)(y - mu0)/(1-e) on x."""
    name = "DR-learner"

    def __init__(self, seed=0, propensity: float | None = None, folds: int = 2):
        self.seed, self.e, self.folds = seed, propensity, folds

    def fit(self, X, t, y):
        X, t, y = np.asarray(X), np.asarray(t), np.asarray(y)
        e = self.e if self.e is not None else t.mean()
        phi = np.zeros(len(y))
        for a, b in StratifiedKFold(self.folds, shuffle=True, random_state=self.seed).split(X, t * 2 + y):
            m1 = _clf(self.seed).fit(X[a][t[a] == 1], y[a][t[a] == 1])
            m0 = _clf(self.seed).fit(X[a][t[a] == 0], y[a][t[a] == 0])
            mu1, mu0 = m1.predict_proba(X[b])[:, 1], m0.predict_proba(X[b])[:, 1]
            phi[b] = mu1 - mu0 + t[b] * (y[b] - mu1) / e - (1 - t[b]) * (y[b] - mu0) / (1 - e)
        self.g = _reg(self.seed).fit(X, phi)
        return self

    def predict(self, X):
        return self.g.predict(X)


class TransformedOutcome:
    """Athey & Imbens (2015): E[Y* | x] = tau(x) for Y* = y (t - e) / (e (1 - e)). Simple but high-variance."""
    name = "Transformed outcome"

    def __init__(self, seed=0, propensity: float | None = None):
        self.seed, self.e = seed, propensity

    def fit(self, X, t, y):
        t, y = np.asarray(t), np.asarray(y)
        e = self.e if self.e is not None else t.mean()
        self.g = _reg(self.seed).fit(X, y * (t - e) / (e * (1 - e)))
        return self

    def predict(self, X):
        return self.g.predict(X)


class ResponseModel:
    """NOT an uplift model: P(y=1 | x, treated). The common industry shortcut ('target likely buyers')."""
    name = "Response model (baseline)"

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, t, y):
        X, t, y = np.asarray(X), np.asarray(t), np.asarray(y)
        self.m = _clf(self.seed).fit(X[t == 1], y[t == 1])
        return self

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1]


class RandomScore:
    name = "Random targeting"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def fit(self, X, t, y):
        return self

    def predict(self, X):
        return self.rng.random(len(X))


ALL_LEARNERS = [SLearner, TLearner, XLearner, DRLearner, TransformedOutcome, ResponseModel, RandomScore]
