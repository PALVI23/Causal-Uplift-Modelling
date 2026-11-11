# Who Should We Target? Causal Uplift Modelling on a 1.4M-User Randomised Advertising Experiment

**Author:** _Your Name_ · **Code:** _github.com/your-username/uplift_ · **Data:** Criteo Uplift Prediction Dataset v2.1 (Diemert et al., 2021)

## Abstract
Targeting models usually predict *who will respond*. A marketer needs *who will respond because of the treatment*:
the conditional average treatment effect (CATE), or uplift. Using a 1.4M-user stratified sample of Criteo's randomised
ad-incrementality test, I (i) analyse the experiment (+0.93 pp visit rate, +24% relative; conversions +47%), (ii) compare five
CATE meta-learners against a response-model baseline using Qini curves with bootstrap CIs, and (iii) use two experiments
with known ground truth. A **semi-synthetic benchmark** with real covariates and simulated outcomes shows the S-learner
recovers individual effects best (PEHE 0.017, Spearman 0.77 with the true effect). The Qini coefficient, which can be computed on real data,
ranks the learners in the same order as these true-effect metrics, which cannot. It also shows **when** a response model works: when
uplift tracks baseline propensity. An **observational-bias experiment** turns the RCT into confounded data. Naive comparison
overstates the effect 4–8×. IPW with the true propensity recovers the randomised truth, but estimated-propensity IPW and AIPW
under-correct, which shows that nuisance estimation is the bottleneck in practice.

## 1. The experiment
1,397,959 users (10% sample, stratified by arm). Randomisation checks pass: treated share 0.8500 vs 0.85 by design
(sample-ratio-mismatch test p = 0.998), and every feature is balanced (max |standardised mean difference| = 0.046 < 0.1).

| Outcome | Control | Treated | ATE (95% CI) | Relative | p |
|---|---|---|---|---|---|
| Visit | 3.885% | 4.817% | **+0.931 pp** [0.840, 1.023] | +24.0% | < 1e-15 |
| Conversion | 0.207% | 0.304% | **+0.097 pp** [0.075, 0.119] | +46.9% | 2e-14 |

**Power.** With this sample the minimum detectable effect at 80% power is 0.128 pp for visits and 0.030 pp for conversions.
Both observed effects are far above it. At the 85/15 split, detecting the observed effects needs about 4.1k control users for visits but
about 21.6k for conversions, because rare outcomes need about 5× more data.

## 2. Uplift models on the real data (holdout: 30%, about 419k users)
Five CATE estimators with LightGBM bases: **S-learner** (one model with *t* as a feature), **T-learner** (one model per arm),
**X-learner** (Künzel et al., 2019), **DR-learner** (Kennedy, 2020; cross-fitted) and **transformed outcome** (Athey & Imbens, 2015).
Baselines: a **response model** P(visit | x, treated), the common industry shortcut, and random targeting.

| Model (visit) | Qini coef. (95% bootstrap CI) | Uplift in top 10% | Uplift in top 30% |
|---|---|---|---|
| **S-learner** | **0.090** [0.076, 0.105] | **6.04 pp** | 2.72 pp |
| DR-learner | 0.081 [0.065, 0.097] | 5.21 pp | 2.62 pp |
| X-learner | 0.080 [0.065, 0.096] | 5.56 pp | 2.67 pp |
| Response model | 0.074 [0.058, 0.090] | 4.26 pp | 2.61 pp |
| Transformed outcome | 0.071 [0.057, 0.089] | 5.09 pp | 2.54 pp |
| T-learner | 0.064 [0.051, 0.082] | 5.43 pp | 2.48 pp |
| Random | 0.004 [−0.007, 0.015] | 1.08 pp | 1.07 pp |

* The best model's top decile has an uplift of **6.0 pp, 6.5× the average effect** (0.93 pp). Ad effects are highly concentrated.
* The S-learner ranks best, but its CI overlaps with the X-/DR-learners and the response model. These marginal bootstrap CIs
  cannot separate them; a paired bootstrap of the differences would be the right test (future work).
* The T-learner is weakest, as expected with a small control arm (15%): its control model is fit on less data and is noisier.

**Targeting policy.** Assumptions: an incremental visit is worth 1.0 and showing ads costs 0.005 per user. The targeting fraction
was chosen on one half of the holdout and evaluated on the other half. Targeting the top **35%** by S-learner uplift earns
**7.14 per 1,000 users vs 4.04 for targeting everyone (+77%)**, compared with 6.31 for the response model's best policy (45%)
and 3.92 for random targeting.

**Conversion (0.3% base rate).** Here the response model ranks first (Qini 0.149 [0.077, 0.238]), ahead of the X-learner (0.111) and
T-learner (0.110). With only about 1,200 conversions in the holdout, all CIs are wide and overlap. The semi-synthetic results explain why a
response model can do well here: see Section 3.

## 3. Semi-synthetic benchmark: scoring against the *true* individual effect
Real individual effects are never observed. I keep the real Criteo covariates (300k users), use a realistic baseline visit model
fit on held-out real control users, and simulate outcomes with a **known** effect τ(x). This is done in two scenarios, each with 3 replications and a
train/evaluate split:
* **heterogeneous:** the effect varies strongly across users and is negative for some;
* **weak:** a small, nearly constant effect on the log-odds scale. On the probability scale it then grows with baseline propensity.

| Heterogeneous scenario | PEHE ↓ | Spearman with true τ ↑ | Qini ↑ |
|---|---|---|---|
| **S-learner** | **0.0168** | **0.77** | **0.168** |
| X-learner | 0.0225 | 0.64 | 0.157 |
| DR-learner | 0.0312 | 0.64 | 0.142 |
| Transformed outcome | 0.0357 | 0.63 | 0.133 |
| T-learner | 0.0503 | 0.51 | 0.110 |
| Response model | – | 0.37 | **−0.066** (worse than random) |

Findings:
1. **The S-learner recovers individual effects best**, with about 3× lower PEHE than the T-learner, matching its first place on the real data.
   With an 85/15 split and a moderate effect, pooling both arms in one regularised model beats fitting the small control arm separately.
2. **The Qini coefficient, computable on real data, ranks the learners in the same order as the true-effect metrics** in the
   heterogeneous scenario, and almost the same in the weak one. That supports using Qini for model selection on real RCT data.
3. **When does the response model work?** In the *weak* scenario it has the best rank correlation with the true effect (0.95)
   and the best Qini, because a constant log-odds effect makes the probability-scale uplift proportional to baseline propensity. In the
   *heterogeneous* scenario it ranks users **worse than random** (Qini −0.066). This explains the real-data conversion result: where uplift
   tracks baseline propensity, "target likely converters" is a strong heuristic; where effects are heterogeneous, it fails.

## 4. What if it had not been an experiment? Observational bias
I sub-sample the RCT so that users with a high baseline visit probability are more likely to be treated, which mimics real campaigns
aimed at "good" customers. The keep-probabilities are chosen so that P(T=1 | x) equals a known propensity s(x) while the
**covariate distribution is unchanged**, so the randomised ATE on the same users remains the ground truth (0.94 pp).

| Estimator: bias in pp (covers truth?) | strength 0.5 | strength 1.0 | strength 1.5 |
|---|---|---|---|
| Naive difference in means | **+3.60** (no) | **+6.02** (no) | **+7.19** (no) |
| Outcome regression | −0.27 (no) | −0.07 (no) | −0.05 (no) |
| IPW, estimated propensity | −0.43 (no) | −0.75 (no) | −1.25 (no) |
| **IPW, true propensity** | **−0.08 (yes)** | **−0.11 (yes)** | **−0.08 (yes)** |
| AIPW / doubly robust | −0.32 (no) | −0.41 (no) | −0.59 (no) |

1. **The naive comparison overstates the effect 3.8×–7.6×.** This is selection bias from targeting likely visitors.
2. **The design is correct:** IPW with the *true* propensity recovers the randomised truth at every strength. In development checks on
   simulated data at the same scale (1.4M users), IPW and AIPW with *estimated* nuisances also covered the truth.
3. **On the real data, estimated-propensity IPW and AIPW under-correct** and their CIs miss the truth, and this gets worse with stronger confounding.
   Outcome regression gets the closest point estimate, but its standard error ignores model uncertainty, so its CI is too narrow.
   Since the true-propensity estimator works, the failure lies in **estimating the nuisance functions** on these features, not in the estimators.
   *Hypothesis (not yet tested):* the anonymised Criteo features are heavily tied and skewed, so the propensity model is mis-calibrated where
   s(x) is extreme, and inverse weighting amplifies those errors. Next steps: calibration plots of ê(x) against s(x), propensity calibration and
   trimming, and overlap-restricted estimands.
4. **Lesson:** "doubly robust" is not "automatically robust". Recovering causal effects from observational data depends on getting
   the propensity model right. Where possible, run the experiment.

## 5. A bug caught during development
The first version of the observational experiment fit the "baseline visit" score on the *same* users it then sub-sampled. The
in-sample model memorised outcome noise, so treatment assignment leaked the outcome and unconfoundedness was violated. Even AIPW with the
**true** outcome model was biased. Splitting the AIPW estimate into its terms exposed it: control residuals had mean −0.0075 instead of 0.
The score is now fit on a separate 20% of users, and a regression test guards against the bug coming back.

## Limitations
* A 10% sample (1.4M of 13.9M users). Features are anonymised, so effects can't be interpreted in business terms.
* The marginal bootstrap CIs don't test model *differences*; a paired bootstrap would.
* The policy economics (value per visit, cost per user) are illustrative assumptions.
* The semi-synthetic results depend on the chosen effect functions. Two scenarios are not exhaustive.
