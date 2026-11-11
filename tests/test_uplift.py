import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uplift import causal, experiment, learners, metrics  # noqa: E402
from uplift.data import FEATURES, make_synthetic  # noqa: E402


@pytest.fixture(scope="module")
def data():
    df = make_synthetic(40_000, seed=1)
    return df[FEATURES].values, df.treatment.values, df.visit.values


def test_qini_matches_sklift(data):
    sk = pytest.importorskip("sklift.metrics")
    X, t, y = data
    s = X[:, 2] + 0.1 * np.random.default_rng(0).normal(size=len(y))
    assert metrics.qini_coefficient(y, t, s) == pytest.approx(sk.qini_auc_score(y, s, t, negative_effect=True), abs=2e-3)
    assert metrics.uplift_at_k(y, t, s, 0.3) == pytest.approx(sk.uplift_at_k(y, s, t, strategy="overall", k=0.3), abs=1e-3)


def test_metric_sanity(data):
    X, t, y = data
    good, rand = X[:, 2], np.random.default_rng(0).random(len(y))
    # Qini coefficients are small in absolute terms (the 'perfect' curve uses realised outcomes)
    assert metrics.qini_coefficient(y, t, good) > 3 * abs(metrics.qini_coefficient(y, t, rand))
    assert metrics.qini_coefficient(y, t, y * t - y * (1 - t)) == pytest.approx(1.0)


def test_learners_rank_effect(data):
    X, t, y = data
    h = len(y) // 2
    q_rand = abs(metrics.qini_coefficient(y[h:], t[h:], np.random.default_rng(0).random(len(y) - h)))
    for L in [learners.TLearner, learners.XLearner, learners.DRLearner, learners.SLearner]:
        tau = L(seed=0).fit(X[:h], t[:h], y[:h]).predict(X[h:])          # held-out evaluation
        assert np.corrcoef(tau, X[h:, 2] > 0)[0, 1] > 0.1, L.name
        assert metrics.qini_coefficient(y[h:], t[h:], tau) > max(0.01, 2 * q_rand), L.name


def test_ab_stats():
    rng = np.random.default_rng(0)
    t = (rng.random(200_000) < 0.85).astype(int)
    y = (rng.random(200_000) < 0.04 + 0.01 * t).astype(int)
    r = experiment.ate_binary(y, t)
    assert r["ci_low"] < 0.01 < r["ci_high"] and r["p_value"] < 1e-6
    assert experiment.srm_test(t, 0.85)["p_value"] > 0.001


def test_confounded_subsample_keeps_x_and_sets_propensity():
    rng = np.random.default_rng(0)
    n = 400_000
    x = rng.normal(size=n)
    t = (rng.random(n) < 0.85).astype(int)
    s = np.clip(1 / (1 + np.exp(-1.5 * x)), 0.1, 0.9)
    keep = causal.confounded_subsample(t, s, 0.85, seed=0)
    assert abs(x[keep].mean() - x.mean()) < 0.02 and abs(x[keep].std() - x.std()) < 0.02
    hi = keep & (x > 1)
    assert abs(t[hi].mean() - s[x > 1].mean()) < 0.02


def test_aipw_recovers_ate_under_confounding():
    rng = np.random.default_rng(1)
    n = 300_000
    X = rng.normal(size=(n, 4))
    t = (rng.random(n) < 0.85).astype(int)
    p = 1 / (1 + np.exp(-(-2.5 + 1.2 * X[:, 0] + 0.5 * t)))
    y = (rng.random(n) < p).astype(int)
    true_ate = y[t == 1].mean() - y[t == 0].mean()
    s = np.clip(1 / (1 + np.exp(-1.5 * X[:, 0])), 0.1, 0.9)
    k = causal.confounded_subsample(t, s, 0.85, seed=2)
    est = causal.ate_estimators(X[k], t[k], y[k], oracle_propensity=s[k])
    assert est.loc["Naive difference in means", "ate"] > true_ate + 0.01     # confounding biases naive upward
    assert abs(est.loc["AIPW / doubly robust", "ate"] - true_ate) < 0.006


def test_confounding_design_has_no_outcome_leakage():
    """Regression test for a bug found during development: if the score that drives the simulated treatment
    assignment is fit on the same users, it memorises their outcomes and AIPW/IPW become biased."""
    df = make_synthetic(400_000, seed=3)
    part = np.random.default_rng(0).random(len(df)) < 0.2
    A, B = df[part], df[~part]
    ca = A[A.treatment == 0]
    base_fn = causal.baseline_logit_model(ca[FEATURES].values, ca.visit.values)
    X = B[FEATURES].values
    z = base_fn(X); z = (z - z.mean()) / z.std()
    truth = experiment.ate_binary(B.visit, B.treatment)["ate"]
    s = np.clip(1 / (1 + np.exp(-1.0 * z)), 0.05, 0.95)
    k = causal.confounded_subsample(B.treatment.values, s, 0.85, seed=0)
    est = causal.ate_estimators(X[k], B.treatment.values[k], B.visit.values[k])
    assert est.loc["Naive difference in means", "ate"] - truth > 0.02
    assert abs(est.loc["AIPW / doubly robust", "ate"] - truth) < 3 * est.loc["AIPW / doubly robust", "se"] + 0.002
