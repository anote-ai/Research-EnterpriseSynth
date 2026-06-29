# How Much Accuracy Do You Give Up for Differential Privacy? A First Look.

*Draft blog post — accompanies the EnterpriseSynth benchmark.*
*Status: based on real baseline measurements; the DP-specific numbers below are
explicitly marked as projected pending a full DP-SGD experiment run (see
RESEARCH_STATUS.md for details).*

---

## The question every compliance officer asks

If your company wants to train a model on synthetic data instead of real
customer records — because the real records are covered by GDPR, HIPAA, or
SOX — there are two questions that come up immediately:

1. How much accuracy will we lose by using synthetic data instead of real data?
2. If we add a formal differential privacy (DP) guarantee on top, how much
   *more* accuracy do we lose?

Most synthetic data vendors will tell you "not much." Almost none of them show
you the curve. EnterpriseSynth's first goal is to actually plot that curve
for enterprise-style tabular data, instead of asserting it.

## What we measured (real numbers, not projections)

We took three public datasets that act as proxies for common enterprise
schemas — Adult Income (HR/CRM proxy), Credit-G (financial transactions
proxy), and Diabetes PIMA (healthcare EHR proxy) — and ran three popular
**non-DP** synthetic data generators against each: GaussianCopula, CTGAN, and
TVAE. For each, we trained a classifier on the synthetic data and tested it
on real held-out data (Train-Synthetic-Test-Real, or TSTR), then compared
that to a model trained directly on real data (the "oracle").

This establishes the **utility ceiling**: the best accuracy you could hope
for from synthetic data, before any privacy mechanism is layered on top.

| Dataset | Best synthesizer | TSTR F1 | Oracle F1 | Utility retained |
|---|---|---|---|---|
| Adult Income (HR proxy) | TVAE | 0.620 | 0.658 | 94.3% |
| Credit-G (financial proxy) | GaussianCopula | 0.824 | 0.797 | 103% (synthetic outperformed oracle F1 here — small dataset, high variance) |
| Diabetes PIMA (healthcare proxy) | GaussianCopula | 0.512 | 0.560 | 91.4% |

(Full numbers, including per-synthesizer Wasserstein distances and runtimes,
are in `results/baseline_sdg.json` — these are real measured outputs of
`scripts/run_baseline_sdg.py`, including wall-clock training time, e.g. CTGAN
took 408 seconds to fit and sample on the Adult dataset.)

**Takeaway:** even without any privacy mechanism, synthetic tabular data
already gives up roughly 5-10% utility relative to training on real data —
and that gap varies a lot by dataset and synthesizer choice. There is no
universal "synthetic data is 95% as good as real data" number; it depends
heavily on what you're synthesizing and which method you use.

## What we have NOT yet measured: the DP cost

The harder, more important question is what happens when you add a formal
(ε, δ)-differential-privacy guarantee via DP-SGD training. This is the
centerpiece of the EnterpriseSynth research design (see DESIGN_DOC.md).

**We have not yet run that experiment with real DP-SGD training.** The
numbers currently published in `paper/draft.md` (Table 2, Appendix B) and
`results/epsilon_sweep.json` are produced by a calibrated simulation function
(`_simulated_scores()` in `scripts/run_epsilon_sweep.py`) that interpolates
plausible DP degradation curves anchored to the real oracle F1 values above —
they are **projections, not measurements**. We are flagging this explicitly
rather than presenting them as results, because:

- The simulation currently produces *identical* curves for HR, financial, and
  healthcare domains, which contradicts the design doc's own hypothesis that
  domain type should change the tradeoff.
- No actual DP-SGD training run (e.g. via Opacus) has been executed for this
  benchmark yet.

So, projected (not measured) expectations for what we'll find once that
experiment runs, based on the literature and the calibration curve:

- At ε = 2 (a HIPAA-compatible privacy budget), utility retention is
  *projected, pending full experiment run* to land around 79-81% of the
  real-data oracle.
- Below ε = 1, utility is *projected* to fall off a cliff, consistent with
  what's been reported for DP-SGD on image and text models.

We will update this post — and replace "projected" with "measured" — once
`scripts/run_epsilon_sweep.py` is wired up to a real DP-SGD synthesizer
(tracked as a follow-up; see RESEARCH_STATUS.md).

## Why this matters

Synthetic data and differential privacy are both routinely sold as drop-in
solutions for "we can't use real data here." The honest answer is that there
is a real, measurable utility cost even before you add formal privacy, and an
additional cost — currently unmeasured for this benchmark — once you do. Until
someone runs and publishes the real DP-SGD numbers (which is the next concrete
step for this project), anyone telling you "you'll barely notice the
difference" is guessing, just like our current placeholder numbers are.

---

*This post accompanies the EnterpriseSynth project
(github.com/anote-ai/Research-EnterpriseSynth). See `paper/draft.md` for the
full technical writeup and `RESEARCH_STATUS.md` for a transparent breakdown of
what is measured vs. projected throughout the repo.*
