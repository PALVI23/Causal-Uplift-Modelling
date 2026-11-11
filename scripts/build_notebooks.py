"""Generates the notebooks in ../notebooks.   Run: python scripts/build_notebooks.py"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

SETUP = '''# ---------- Setup: run this cell first (CPU runtime is enough) ----------
import os, sys, subprocess, json, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
SMOKE = os.environ.get("UP_SMOKE") == "1"          # local smoke test on synthetic data
try:
    import google.colab  # noqa: F401
    ON_COLAB = True
except ImportError:
    ON_COLAB = False
if SMOKE:
    PROJECT_DIR = os.environ["UP_PROJECT_DIR"]
elif ON_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    import glob
    hits = sorted(glob.glob("/content/drive/MyDrive/**/uplift/src/uplift/data.py", recursive=True), key=len)
    assert hits, "Upload the 'uplift' folder to your Google Drive first."
    PROJECT_DIR = str(Path(hits[0]).parents[2])                     # found automatically, wherever you put it
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabulate"], check=True)
else:   # local Jupyter / VS Code: notebook lives in uplift/notebooks/
    PROJECT_DIR = str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd())
sys.path.insert(0, f"{PROJECT_DIR}/src")
ART, FIG = f"{PROJECT_DIR}/artifacts", f"{PROJECT_DIR}/reports/figures"
DATA = "/content/data" if ON_COLAB else f"{PROJECT_DIR}/data"
for d in (ART, FIG, DATA): os.makedirs(d, exist_ok=True)

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid"); plt.rcParams["figure.dpi"] = 110
from uplift.data import FEATURES, load_criteo, make_synthetic
from uplift import experiment as E, learners as L, metrics as M, causal as C
SEED = 42
df = make_synthetic(60_000) if SMOKE else load_criteo(DATA, sample_frac=0.10)   # 10% sample ~ 1.4M users
TREAT_SHARE = 0.85                                                              # by design of the experiment
def savefig(name):
    plt.tight_layout(); plt.savefig(f"{FIG}/{name}.png", dpi=160, bbox_inches="tight"); plt.show()
def save_json(o, name):
    json.dump(o, open(f"{ART}/{name}.json", "w"), indent=2, default=float)
def load_json(name):
    return json.load(open(f"{ART}/{name}.json"))
print(f"{len(df):,} users | treated share {df.treatment.mean():.3f} | visit {df.visit.mean():.4f} | conversion {df.conversion.mean():.5f}")'''


def nb(cells):
    book = nbf.v4.new_notebook()
    book.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
    for kind, src in cells:
        src = src.strip("\n")
        book.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    return book


NB1 = [
("md", """# 01 · Analysing the randomised experiment
**Project:** *Who should we target?* Causal uplift modelling on the Criteo randomised ad experiment.
**Data:** [Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) (Diemert et al., 2021),
13.9M users. 85% were eligible to see ads (treatment) and 15% were held out (control). Outcomes: site **visit** and **conversion**.
We work with a stratified 10% sample (about 1.4M users), downloaded automatically from Hugging Face (`criteo/criteo-uplift`)
or Kaggle (`arashnic/uplift-modeling`, which uses your `KAGGLE_API_TOKEN` Colab secret). **Runtime: CPU is enough** (about 10 min the first time)."""),
("code", SETUP),
("md", "## 1. Is the randomisation healthy? Sample-ratio mismatch and covariate balance"),
("code", '''srm = E.srm_test(df.treatment, TREAT_SHARE); print(srm)
bal = E.balance_table(df, FEATURES)
print(f"max |SMD| = {bal.smd.abs().max():.4f}  (< 0.1 = balanced)")
fig, ax = plt.subplots(figsize=(6, 3.5)); bal.smd.plot.bar(ax=ax, color="#4C72B0")
ax.axhline(0.1, ls="--", c="r", lw=1); ax.axhline(-0.1, ls="--", c="r", lw=1); ax.set_title("Covariate balance (standardised mean difference)")
savefig("01_balance"); bal.round(4)'''),
("md", "## 2. Average treatment effect (ATE): does advertising work on average?"),
("code", '''ate = {y: E.ate_binary(df[y], df.treatment) for y in ["visit", "conversion"]}
ate_df = pd.DataFrame(ate).T
ate_df[["rate_treated", "rate_control", "ate", "ci_low", "ci_high", "relative_lift", "p_value"]]'''),
("code", '''fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for a, y in zip(ax, ["visit", "conversion"]):
    r = ate[y]
    a.bar(["control", "treated"], [r["rate_control"], r["rate_treated"]], color=["#BBBBBB", "#C44E52"])
    a.set_title(f"{y}: +{r['ate']*100:.3f} pp ({r['relative_lift']:.0%} relative)")
savefig("01_ate")'''),
("md", """## 3. Power: how much data does each outcome need?
Conversions are rare (about 0.3%). How small an effect can this sample detect, and how many users would a conversion test need?"""),
("code", '''n1, n0 = int(df.treatment.sum()), int((1 - df.treatment).sum())
power = []
for y in ["visit", "conversion"]:
    p0 = ate[y]["rate_control"]
    power.append({"outcome": y, "control_rate": p0, "observed_ate": ate[y]["ate"],
                  "MDE (80% power)": E.mde(p0, n1, n0),
                  "control users needed for observed effect": E.required_n_per_group(p0, ate[y]["ate"], ratio=n1 / n0)})
power = pd.DataFrame(power).set_index("outcome"); power'''),
("md", "## 4. A first look at heterogeneity: is the effect the same for everyone?"),
("code", '''fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharey=True)
het = {}
for ax, f in zip(axes.ravel(), FEATURES):
    h = E.effect_by_quantile(df, f, "visit", q=5); het[f] = h       # value-based bins (features have many ties)
    if len(h):
        ax.errorbar(range(len(h)), h.ate, yerr=[h.ate - h.ci_low, h.ci_high - h.ate], fmt="o-", capsize=3)
        ax.set_xticks(range(len(h)), h.index)
    ax.axhline(ate["visit"]["ate"], c="r", ls="--", lw=1); ax.set_title(f"{f} ({len(h)} bins)")
fig.suptitle("Visit uplift by feature quintile (red = overall ATE; fewer bins = heavily tied feature)"); savefig("01_heterogeneity")
spread = pd.Series({f: (h.ate.max() - h.ate.min()) if len(h) > 1 else 0.0 for f, h in het.items()}).sort_values(ascending=False)
spread.to_frame("max - min bin ATE").head(6)'''),
("code", '''save_json({"srm": srm, "max_abs_smd": float(bal.smd.abs().max()), "ate": ate, "power": power.reset_index().to_dict(orient="records"),
           "n": len(df), "heterogeneity_spread_top": spread.head(5).to_dict()}, "experiment")
print("saved")'''),
("md", """## Observations (fill in yourself)
* Is the randomisation valid?
* How big is the effect on visits and on conversions, and is it statistically significant?
* Which features suggest heterogeneity? Why does that motivate uplift modelling?"""),
]

NB2 = [
("md", """# 02 · Uplift modelling: estimating *individual* treatment effects
Five ways to estimate the effect for each user (CATE), compared with two baselines: a **response model**
(the common shortcut of targeting likely visitors) and **random** targeting. All models are trained on 70% of users
and evaluated on a 30% holdout with **Qini / uplift curves**, the standard way to evaluate uplift models on randomised data,
with bootstrap confidence intervals. **Runtime: CPU**, about 20–30 min."""),
("code", SETUP),
("code", '''from sklearn.model_selection import train_test_split
tr, te = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df.treatment * 2 + df.visit)
Xtr, Xte = tr[FEATURES].values, te[FEATURES].values
OUTCOME = "visit"
models, scores, fit_min = {}, {}, {}
for Cls in L.ALL_LEARNERS:
    t0 = time.time()
    kw = {"propensity": TREAT_SHARE} if Cls in (L.XLearner, L.DRLearner, L.TransformedOutcome) else {}
    m = Cls(seed=SEED, **kw).fit(Xtr, tr.treatment.values, tr[OUTCOME].values)
    models[m.name], scores[m.name] = m, m.predict(Xte); fit_min[m.name] = (time.time() - t0) / 60
    print(f"{m.name:28s} {fit_min[m.name]:.1f} min")'''),
("md", "## 1. Qini curves (holdout)"),
("code", '''yte, tte = te[OUTCOME].values, te.treatment.values
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for name, s in scores.items():
    x, q = M.qini_curve(yte, tte, s); step = max(1, len(x) // 400)
    ax[0].plot(x[::step], q[::step], label=f"{name} ({M.qini_coefficient(yte, tte, s):.3f})",
               ls="--" if "Random" in name else "-")
    x, u = M.uplift_curve(yte, tte, s)
    ax[1].plot(x[::step], u[::step], label=name, ls="--" if "Random" in name else "-")
ax[0].set_title("Qini curve (legend: Qini coefficient)"); ax[0].set_xlabel("fraction of users targeted"); ax[0].set_ylabel("incremental visits")
ax[1].set_title("Uplift curve"); ax[1].set_xlabel("fraction of users targeted")
ax[0].legend(fontsize=8); savefig("02_qini_curves")'''),
("md", "## 2. Model comparison with bootstrap 95% CIs"),
("code", '''NB = 30 if SMOKE else 100
rows = []
for name, s in scores.items():
    mean, lo, hi = M.bootstrap(yte, tte, s, M.qini_coefficient, n_boot=NB, seed=SEED)
    rows.append({"model": name, **M.summary(yte, tte, s), "qini_ci_low": lo, "qini_ci_high": hi, "fit_minutes": fit_min[name]})
comp = pd.DataFrame(rows).set_index("model").sort_values("qini_coef", ascending=False)
comp.to_csv(f"{ART}/model_comparison_{OUTCOME}.csv"); comp.round(4)'''),
("code", '''fig, ax = plt.subplots(figsize=(7, 3.8))
c = comp.sort_values("qini_coef")
ax.barh(c.index, c.qini_coef, xerr=[c.qini_coef - c.qini_ci_low, c.qini_ci_high - c.qini_coef], color="#4C72B0", capsize=3)
ax.axvline(0, c="k", lw=1); ax.set_xlabel("Qini coefficient (95% bootstrap CI)"); ax.set_title(f"Holdout ranking quality ({OUTCOME})")
savefig("02_model_comparison")'''),
("md", "## 3. Does the ranking hold up? Observed uplift by predicted-uplift decile (best model)"),
("code", '''BEST = comp.drop(index=[i for i in comp.index if "Random" in i or "Response" in i]).index[0]
dec = M.decile_table(yte, tte, scores[BEST]); dec_resp = M.decile_table(yte, tte, scores["Response model (baseline)"])
fig, ax = plt.subplots(figsize=(8, 3.8)); w = 0.4
ax.bar(np.arange(10) - w/2, dec.observed_uplift, w, label=BEST); ax.bar(np.arange(10) + w/2, dec_resp.observed_uplift, w, label="Response model")
ax.axhline(ate_all := yte[tte == 1].mean() - yte[tte == 0].mean(), c="r", ls="--", lw=1, label="average effect")
ax.set_xticks(range(10), range(1, 11)); ax.set_xlabel("decile of predicted score (1 = top)"); ax.set_ylabel("observed uplift")
ax.legend(); ax.set_title("Observed treatment effect per decile"); savefig("02_deciles"); dec.round(4)'''),
("md", """## 4. From scores to a decision: profit-optimal targeting
Illustrative economics (stated assumptions): each incremental visit is worth **v** and showing ads to a user costs **c**.
We choose the targeting fraction on one half of the holdout and report the profit on the other half, to avoid optimistic bias."""),
("code", '''VALUE, COST = 1.00, 0.005          # value per incremental visit, cost per targeted user (assumptions)
half = np.random.default_rng(SEED).random(len(yte)) < 0.5
pol = {}
for name in [BEST, "Response model (baseline)", "Random targeting"]:
    sel = M.policy_curve(yte[half], tte[half], scores[name][half], VALUE, COST)
    k = sel.loc[sel.profit_per_1000.idxmax(), "target_frac"]
    ev = M.policy_curve(yte[~half], tte[~half], scores[name][~half], VALUE, COST, grid=[k, 1.0])
    pol[name] = {"chosen_target_frac": k, "profit_per_1000 (chosen)": ev.profit_per_1000.iloc[0],
                 "profit_per_1000 (target everyone)": ev.profit_per_1000.iloc[1]}
pol = pd.DataFrame(pol).T; pol.to_csv(f"{ART}/policy.csv"); pol.round(3)'''),
("code", '''fig, ax = plt.subplots(figsize=(7, 4))
for name in [BEST, "Response model (baseline)", "Random targeting"]:
    pc = M.policy_curve(yte, tte, scores[name], VALUE, COST)
    ax.plot(pc.target_frac, pc.profit_per_1000, marker="o", ms=3, label=name)
ax.axhline(0, c="k", lw=1); ax.set_xlabel("fraction of users targeted"); ax.set_ylabel("profit per 1,000 users")
ax.set_title(f"Targeting policy (value/visit={VALUE}, cost/user={COST})"); ax.legend(); savefig("02_policy")'''),
("md", "## 5. Secondary outcome: conversion (rare, about 0.3%). Best two learners vs baselines"),
("code", '''conv = {}
for Cls in [L.XLearner, L.DRLearner, L.TLearner, L.ResponseModel, L.RandomScore]:
    kw = {"propensity": TREAT_SHARE} if Cls in (L.XLearner, L.DRLearner) else {}
    m = Cls(seed=SEED, **kw).fit(Xtr, tr.treatment.values, tr.conversion.values)
    s = m.predict(Xte)
    conv[m.name] = {**M.summary(te.conversion.values, tte, s), **dict(zip(["qini_boot_mean", "qini_ci_low", "qini_ci_high"],
                    M.bootstrap(te.conversion.values, tte, s, n_boot=NB, seed=SEED)))}
conv = pd.DataFrame(conv).T.sort_values("qini_coef", ascending=False); conv.to_csv(f"{ART}/model_comparison_conversion.csv"); conv.round(4)'''),
("code", '''save_json({"best_model": BEST, "outcome": OUTCOME, "n_train": len(tr), "n_test": len(te), "value": VALUE, "cost": COST}, "modeling")
print("saved")'''),
("md", """## Observations (fill in yourself)
* Which learner ranks best, and are the differences significant (overlapping CIs)?
* Why does the response model do worse than the uplift models, if it does?
* What does the decile plot tell you about who is *persuadable* and who would visit anyway?"""),
]

NB3 = [
("md", """# 03 · Experiments with known ground truth
On real data we never observe an individual's treatment effect. Two experiments get around this:
1. **Semi-synthetic benchmark:** real Criteo covariates and a realistic baseline, with *simulated* outcomes whose individual
   effect is **known**. Learners can then be scored on PEHE (RMSE of the individual effect), not only on ranking.
2. **Observational bias:** we turn the RCT into *confounded* observational data, where users likely to visit anyway are more
   likely to be treated, and test which estimators recover the true (randomised) average effect.

**Runtime: CPU**, about 20–30 min."""),
("code", SETUP),
("md", "## 1. Semi-synthetic benchmark"),
("code", '''N_SS = 20_000 if SMOKE else 300_000
# A realistic baseline-visit model, fit on a SEPARATE 20% of users. Using the same users later would leak their
# own outcomes into the simulated treatment assignment (in-sample fit memorises noise) and break unconfoundedness.
fit_part = np.random.default_rng(SEED).random(len(df)) < 0.2
dfA, dfB = df[fit_part], df[~fit_part].reset_index(drop=True)
ctrlA = dfA[dfA.treatment == 0]
base_fn = C.baseline_logit_model(ctrlA[FEATURES].values, ctrlA.visit.values, seed=SEED)
ss = dfB.sample(min(N_SS, len(dfB)), random_state=SEED).reset_index(drop=True)
X = ss[FEATURES].values
base = base_fn(X)
SS_LEARNERS = [L.SLearner, L.TLearner, L.XLearner, L.DRLearner, L.TransformedOutcome, L.ResponseModel]
rows = []
for scenario in ["heterogeneous", "weak"]:
    for rep in range(1 if SMOKE else 3):
        t, y, tau = C.semi_synthetic(X, base, scenario, TREAT_SHARE, seed=rep)
        h = len(y) // 2                       # train on first half, evaluate on second
        for Cls in SS_LEARNERS:
            kw = {"propensity": TREAT_SHARE} if Cls in (L.XLearner, L.DRLearner, L.TransformedOutcome) else {}
            m = Cls(seed=rep, **kw).fit(X[:h], t[:h], y[:h]); th = m.predict(X[h:])
            rows.append({"scenario": scenario, "rep": rep, "model": m.name,
                         "pehe": C.pehe(th, tau[h:]) if "Response" not in m.name else np.nan,
                         "rank_corr_with_true_tau": pd.Series(th).corr(pd.Series(tau[h:]), method="spearman"),
                         "qini_coef": M.qini_coefficient(y[h:], t[h:], th),
                         "ate_error": (th.mean() - tau[h:].mean()) if "Response" not in m.name else np.nan})
        print(scenario, rep, "done")
bench = pd.DataFrame(rows)
bench_sum = bench.groupby(["scenario", "model"])[["pehe", "rank_corr_with_true_tau", "qini_coef", "ate_error"]].agg(["mean", "std"])
bench.to_csv(f"{ART}/semi_synthetic_runs.csv", index=False); bench_sum.round(4)'''),
("code", '''fig, ax = plt.subplots(1, 2, figsize=(13, 4))
d = bench[bench.scenario == "heterogeneous"]
sns.barplot(data=d[~d.model.str.contains("Response")], x="pehe", y="model", ax=ax[0], color="#4C72B0", errorbar="sd")
ax[0].set_title("PEHE: error in individual effects (lower is better)")
sns.barplot(data=d, x="rank_corr_with_true_tau", y="model", ax=ax[1], color="#55A868", errorbar="sd")
ax[1].set_title("Spearman correlation with the TRUE effect"); ax[1].set_ylabel("")
savefig("03_semi_synthetic")'''),
("code", '''# Does the Qini coefficient (computable on real data) agree with the true-effect metrics (not computable)?
agree = bench.groupby(["scenario", "model"])[["qini_coef", "rank_corr_with_true_tau"]].mean()
print("Spearman(Qini, true rank-corr) across models:",
      round(agree.groupby(level=0).apply(lambda g: g.qini_coef.corr(g.rank_corr_with_true_tau, method="spearman")).mean(), 3))
agree.round(4)'''),
("md", """## 2. Observational bias: what if this had not been an experiment?
We sub-sample the RCT so that **users with a high baseline visit probability are more likely to be treated**, which is
typical of real campaigns that target "good" users. The sub-sampling is designed so that the covariate distribution is
unchanged, so the true ATE equals the randomised estimate on the same users. The model that defines "baseline visit
probability" is fit on a separate 20% of users, to avoid leaking outcomes into treatment assignment."""),
("code", '''Xall = dfB[FEATURES].values                       # users NOT used to fit base_fn
z = base_fn(Xall); z = (z - z.mean()) / z.std()
truth = E.ate_binary(dfB.visit, dfB.treatment)
print(f"Randomised (truth) ATE = {truth['ate']:.5f}  [{truth['ci_low']:.5f}, {truth['ci_high']:.5f}]")
obs_rows = []
for strength in [0.5, 1.0, 1.5]:
    s = np.clip(1 / (1 + np.exp(-strength * z)), 0.05, 0.95)       # true propensity in the confounded sample
    keep = C.confounded_subsample(dfB.treatment.values, s, TREAT_SHARE, seed=SEED)
    est = C.ate_estimators(Xall[keep], dfB.treatment.values[keep], dfB.visit.values[keep], oracle_propensity=s[keep], seed=SEED)
    est["confounding_strength"], est["n"] = strength, int(keep.sum())
    est["bias"] = est.ate - truth["ate"]; est["covers_truth"] = (est.ci_low <= truth["ate"]) & (truth["ate"] <= est.ci_high)
    obs_rows.append(est.reset_index())
    print(f"strength={strength}: n={keep.sum():,}  naive bias={est.loc['Naive difference in means','bias']:+.5f}  "
          f"AIPW bias={est.loc['AIPW / doubly robust','bias']:+.5f}")
obs = pd.concat(obs_rows); obs.to_csv(f"{ART}/observational_bias.csv", index=False)
obs.pivot(index="estimator", columns="confounding_strength", values="bias").round(5)'''),
("code", '''fig, ax = plt.subplots(figsize=(8, 4.5))
for i, (name, g) in enumerate(obs.groupby("estimator", sort=False)):
    ax.errorbar(g.confounding_strength + (i - 2) * 0.05, g.ate, yerr=[g.ate - g.ci_low, g.ci_high - g.ate], fmt="o", capsize=3, label=name)
ax.axhline(truth["ate"], c="k", ls="--", label="randomised truth"); ax.axhspan(truth["ci_low"], truth["ci_high"], color="grey", alpha=.2)
ax.set_xlabel("confounding strength"); ax.set_ylabel("estimated ATE (visit)"); ax.legend(fontsize=8)
ax.set_title("Recovering the experimental effect from confounded data"); savefig("03_observational_bias")'''),
("md", "## 3. Auto-generated results summary (uses notebooks 01–03)"),
("code", '''ex, mo = load_json("experiment"), load_json("modeling")
comp = pd.read_csv(f"{ART}/model_comparison_visit.csv", index_col=0)
conv = pd.read_csv(f"{ART}/model_comparison_conversion.csv", index_col=0)
pol = pd.read_csv(f"{ART}/policy.csv", index_col=0)
v, cv = ex["ate"]["visit"], ex["ate"]["conversion"]
lines = ["# Results summary (auto-generated by notebook 03)", "",
 f"* Data: {ex['n']:,} users (10% stratified sample of Criteo Uplift v2.1); treated share {ex['srm']['share_treated']:.4f} "
 f"(SRM p = {ex['srm']['p_value']:.3f}); max |SMD| = {ex['max_abs_smd']:.4f}",
 f"* **Visit ATE: {v['rate_control']:.4%} → {v['rate_treated']:.4%} = +{v['ate']*100:.3f} pp "
 f"[{v['ci_low']*100:.3f}, {v['ci_high']*100:.3f}], relative +{v['relative_lift']:.1%}, p = {v['p_value']:.1e}**",
 f"* Conversion ATE: +{cv['ate']*100:.4f} pp [{cv['ci_low']*100:.4f}, {cv['ci_high']*100:.4f}], relative +{cv['relative_lift']:.1%}",
 f"* Best uplift model (visit): **{mo['best_model']}**, Qini coefficient {comp.loc[mo['best_model'],'qini_coef']:.4f} "
 f"[{comp.loc[mo['best_model'],'qini_ci_low']:.4f}, {comp.loc[mo['best_model'],'qini_ci_high']:.4f}] vs response model "
 f"{comp.loc['Response model (baseline)','qini_coef']:.4f}",
 f"* Uplift in top 10% (best model): {comp.loc[mo['best_model'],'uplift@10%']*100:.3f} pp vs average {v['ate']*100:.3f} pp",
 "", "## Experiment", pd.DataFrame(ex["ate"]).T[["rate_treated","rate_control","ate","ci_low","ci_high","relative_lift","p_value"]].to_markdown(),
 "", "## Power", pd.DataFrame(ex["power"]).to_markdown(index=False),
 "", "## Uplift models: visit (holdout)", comp.round(4).to_markdown(),
 "", "## Uplift models: conversion (holdout)", conv.round(4).to_markdown(),
 "", f"## Targeting policy (value/visit={mo['value']}, cost/user={mo['cost']})", pol.round(3).to_markdown(),
 "", "## Semi-synthetic benchmark (mean ± sd over reps)", bench_sum.round(4).to_markdown(),
 "", "## Observational bias", obs.round(5).to_markdown(index=False)]
open(f"{PROJECT_DIR}/reports/results_summary.md", "w").write("\\n".join(lines)); print("\\n".join(lines[:8]))'''),
("md", """## Observations (fill in yourself)
* Which learner recovers individual effects best when the truth is known? Does that match the real-data Qini ranking?
* How large is the naive bias under confounding, and which estimators fix it? Do their CIs cover the truth?
* What does this imply for evaluating marketing campaigns that were *not* randomised?"""),
]


def main():
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, cells in [("01_experiment_analysis", NB1), ("02_uplift_modeling", NB2), ("03_ground_truth_experiments", NB3)]:
        nbf.write(nb(cells), out / f"{name}.ipynb")
        print("wrote", name)


if __name__ == "__main__":
    main()
