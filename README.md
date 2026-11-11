# Who should we target? Causal uplift modelling on a 13.9M-user randomised experiment

Most "targeting" models predict **who will buy**. A marketer needs to know **who will buy *because* of the ad**:
the individual treatment effect, or *uplift*. This project estimates it on the
[Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) (Diemert et al., 2021), a real
randomised ad-incrementality test (85% treatment / 15% control; outcomes: visit, conversion). It also asks a second question:
**what goes wrong when the data are *not* randomised?**

## What's inside
1. **Experiment analysis:** sample-ratio-mismatch test, covariate balance (SMD), average treatment effect with CIs, power and MDE for rare outcomes, and subgroup heterogeneity.
2. **Uplift modelling:** S-, T-, X-, DR-learners and transformed-outcome regression with LightGBM bases, against a *response model* (the industry shortcut) and random targeting.
   They are evaluated on a holdout with Qini/uplift curves, AUUC and uplift@k, with bootstrap CIs, plus a decile analysis and a profit-optimal targeting policy chosen on one half of the holdout and evaluated on the other.
3. **Semi-synthetic benchmark:** real Criteo covariates with simulated outcomes whose individual effect is *known*. Learners are scored on PEHE and on rank correlation with the true effect, which also tests whether the Qini coefficient you can compute on real data tracks the true-effect metrics you can't.
4. **Observational bias:** the RCT is turned into *confounded* data by a sub-sampling design that keeps the covariate distribution fixed, so the randomised ATE stays the ground truth.
   It compares naive, outcome-regression, IPW and doubly-robust (AIPW, cross-fitted) estimates and their CI coverage.

## Results (1,397,959 users; full write-up in [`reports/report.md`](reports/report.md))
* **Experiment:** ads raise the visit rate from 3.89% to 4.82%, **+0.93 pp [0.84, 1.02] (+24%)**, and conversions by +47%. Randomisation checks pass (SRM p = 0.998, max |SMD| = 0.046).
* **Uplift is concentrated:** the best model's (S-learner) top 10% has **6.0 pp uplift, 6.5× the average**. Qini 0.090 [0.076, 0.105] vs 0.074 for a response model.
* **Targeting the top 35% by uplift earns +77% profit** compared with targeting everyone (chosen and evaluated on separate halves of the holdout).
* **Semi-synthetic benchmark (true effects known):** the S-learner has the lowest error in individual effects (PEHE 0.017, Spearman 0.77). The Qini coefficient ranks learners in the same order as the true-effect metrics. Response models work only when uplift tracks baseline propensity, and rank **worse than random** when effects are heterogeneous.
* **Without randomisation:** naive comparison overstates the effect **4–8×**. IPW with the true propensity recovers the truth, but estimated-propensity IPW and AIPW under-correct on these features, so in practice the propensity model is the bottleneck.

| Model (visit, holdout) | Qini (95% CI) | Uplift top 10% |
|---|---|---|
| S-learner | **0.090** [0.076, 0.105] | **6.04 pp** |
| DR-learner | 0.081 [0.065, 0.097] | 5.21 pp |
| X-learner | 0.080 [0.065, 0.096] | 5.56 pp |
| Response model (baseline) | 0.074 [0.058, 0.090] | 4.26 pp |
| T-learner | 0.064 [0.051, 0.082] | 5.43 pp |
| Random | 0.004 | 1.08 pp |

## A methodological note (a bug caught by the tests)
The confounding experiment needs a "baseline visit probability" score to drive treatment assignment. A first version fit that score on the *same* users it then sub-sampled. The in-sample model partly memorises each user's own outcome, so selection leaked outcome noise into treatment. That violates unconfoundedness, and even AIPW with the *true* outcome model stayed biased. Decomposing the AIPW terms exposed it: control residuals had mean −0.0075 instead of 0. The fix fits the score on a separate 20% of users, and a regression test (`test_confounding_design_has_no_outcome_leakage`) guards it.