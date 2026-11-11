# Application materials

## Project 1: Credit default risk

### Resume entry (pick 3–4 bullets)
**Calibrated, Explainable & Fair Credit-Risk Modelling** | Python, LightGBM, XGBoost, CatBoost, Optuna, SHAP, Gradio | _[GitHub link]_
- Built an end-to-end credit-default model on 307K loan applications, engineering 242 features from 7 relational tables; reached **0.793 holdout AUC** (Gini 0.59) vs. a 0.774 logistic-regression baseline.
- Ran ablations showing the relational history features added +0.023 AUC, versus < 0.001 between XGBoost, LightGBM and CatBoost, and that class re-weighting worsened calibration 20× (ECE 0.054 vs 0.003).
- Turned the model's probabilities into a profit-maximising approval policy. The empirical threshold (0.165) matched decision theory (0.167) and cut defaults among approved loans by 38% at 87% approval.
- Audited fairness by gender and age. Showed that removing gender fails because other features predict it with AUC 0.90, and that equal-opportunity thresholds close 89% of the gender false-rejection gap for a 0.55% profit cost.
- Generated plain-language SHAP reason codes for rejected applicants and shipped an interactive Gradio demo plus a model card.

**One-line version (if space is tight):**
- Credit-risk ML system (307K loans): 0.793 AUC with LightGBM; calibrated PDs drive a profit-optimal policy (−38% defaults); fairness audit showing gender proxies (AUC 0.90) and a 0.55% cost of equal opportunity.

### SOP paragraph (adapt to your voice; a draft, not final text)
> At work I build AI systems that are judged by whether they run reliably, not by a metric on a leaderboard. To see how far
> rigorous data science goes beyond a good AUC, I built a credit-risk system on 307,000 loan applications. The model itself was
> the easy part: three gradient-boosting libraries finished within 0.001 AUC of each other, while careful relational feature
> engineering added twenty times that. The harder questions came after. Once the probabilities were calibrated, the
> profit-optimal approval threshold I found empirically matched the decision-theoretic value almost exactly. Then the fairness
> audit surprised me: I had removed gender from the model, yet the remaining features predicted it with 0.90 AUC, and
> creditworthy men were still rejected 1.5 times as often as women. Measuring what it would cost to fix this (about half a
> percent of profit) made it clear to me that fairness, causality and decision-making under uncertainty are the questions I
> want to study formally. That is why [Program]'s courses in [X] and [Y], and Professor [Z]'s work on [topic], appeal to me.

### Interview / SOP-review questions you should be able to answer
1. **Why AUC and not accuracy?** With an 8% default rate, approving everyone is 92% "accurate". AUC measures ranking independently of the threshold, and the threshold is chosen separately from costs.
2. **What is calibration and why did it matter here?** Predicted 10% PDs should default about 10% of the time. The break-even rule PD < m/(m+LGD) only works if PDs are calibrated, and the empirical threshold matching theory (0.165 vs 0.167) is the evidence that they are.
3. **Why did class weighting hurt?** It shifts the implied base rate, so the probabilities are no longer calibrated (20× higher ECE). The AUC drop was larger than expected; my hypothesis is that early stopping on the weighted log-loss picked a worse stopping point. I'd test this by early-stopping on AUC instead.
4. **How did you avoid leakage?** The aggregates use only each applicant's own history from before the application. The holdout was split off once and used once. Calibration, threshold, blend weights and fairness thresholds were all fit on out-of-fold predictions.
5. **Why was isotonic calibration worse than Platt?** The model was already nearly calibrated. Isotonic has many more degrees of freedom (a step function), so it slightly overfits the OOF predictions; Platt has two parameters.
6. **Explain equal opportunity vs. demographic parity.** Demographic parity means equal approval rates. Equal opportunity means equal approval rates among *creditworthy* people (equal false-rejection rates). With different base rates (7% vs 10% default), you generally can't satisfy both together with calibration (Kleinberg et al. / Chouldechova impossibility results).
7. **Is your mitigation legal?** Group-specific thresholds use the protected attribute at decision time, which is disparate treatment and generally prohibited in lending (e.g. under the US ECOA). I used them to *measure* the cost of fairness. Lawful routes are removing proxies, in-training constraints and better data for thin-file groups.
8. **What would you do with more time?** Reject inference for the selection bias, early-stopping on AUC for the weighted model, in-training fairness constraints (fairlearn reductions), drift monitoring, and a cost model that depends on loan size.
9. **Why is the model worse for under-30s and over-60s?** Thinner credit histories mean less signal. It's a data-coverage problem as well as a risk difference.
10. **What does a SHAP value mean?** The feature's contribution to this prediction (in log-odds) relative to the average prediction, based on Shapley values from cooperative game theory. It explains the model, not causality.

## Project 3: Text-to-SQL QLoRA
_To be filled once the notebook 03 results are in._


## Project 4: Causal uplift modelling (Criteo)

### Resume entry
**Causal Uplift Modelling on a 1.4M-User Randomised Ad Experiment** | *Python, LightGBM, causal ML (meta-learners, IPW/AIPW), A/B testing* | _[GitHub link]_
- Analysed a randomised ad-incrementality experiment (1.4M users, Criteo): sample-ratio-mismatch and balance checks, ATE with CIs and power analysis. Ads raised visits **+0.93 pp (+24%)** and conversions **+47%**.
- Built and compared **5 CATE estimators** (S/T/X/DR-learners, transformed outcome) with Qini curves and bootstrap CIs. Users in the top uplift decile responded **6.5× more** than average, and uplift-based targeting gave **+77% profit** over targeting everyone.
- Designed a **semi-synthetic benchmark** with known individual effects. Showed that the Qini coefficient ranks models in the same order as the true-effect error, and that response models rank **worse than random** when effects are heterogeneous.
- Simulated confounded observational data from the RCT. Naive estimates were biased **4–8×**; IPW with the true propensity recovered the truth, while estimated-propensity IPW/AIPW under-corrected. Found and fixed an outcome-leakage bug with a decomposition of the AIPW terms, and added a regression test.

**One-line version:** Causal uplift modelling on a 1.4M-user RCT: 5 meta-learners evaluated with Qini curves (top decile 6.5× average uplift, +77% targeting profit), a semi-synthetic benchmark with known effects, and a confounding study showing naive estimates off by 4–8×.

### Interview questions to prepare
1. **Why not just predict who will visit?** That targets "sure things", who visit anyway. Uplift targets "persuadables". The response model only works when uplift tracks baseline propensity (as in the weak scenario); with heterogeneous effects it was worse than random.
2. **How do you evaluate uplift models without ground truth?** Qini and uplift curves on randomised holdout data compare treated and control response rates within each top-k segment. The semi-synthetic benchmark showed Qini ranks models in the same order as the true-effect metrics.
3. **Why did the S-learner win?** With an 85/15 split, the T-learner fits the 15% control arm alone and is noisy. The S-learner pools both arms with regularisation, which helps when effects are moderate. (Its known weakness is shrinking effects towards zero when treatment is a weak feature.)
4. **What is doubly robust?** AIPW is consistent if *either* the outcome model or the propensity model is correct, and it has favourable variance. In practice both are estimated, and errors in the propensity get amplified by inverse weights. Our real-data result showed exactly that.
5. **What was the leakage bug?** The score used to simulate confounding was fit on the same users, so it memorised their outcome noise and treatment depended on the outcome. I found it because AIPW with the *true* outcome model was still biased, and the control residuals had a non-zero mean.
6. **Sample-ratio mismatch?** A χ² test that the treated share matches the design. A failure signals broken randomisation or logging, and invalidates the test.
