# What Happens When You Train AI on AI-Generated Data? We Measured Model Collapse.

*Published on Towards Data Science | estimated read time: 8 minutes*

---

Here's a scenario that's playing out quietly in enterprise AI teams right now.

Your organization can't train models on real customer data — it's too sensitive,
too regulated, or too risky to expose. So you generate synthetic data, train your model
on it, generate more synthetic data using the trained model, and repeat.

Reasonable. Practical. And, according to our benchmark, potentially catastrophic for
any model that needs to detect rare, high-stakes events.

We ran this experiment. We measured what happens. The results should change how every
enterprise team manages synthetic data pipelines.

---

## The Setup

We built a synthetic enterprise dataset with a realistic long-tail distribution:
the kind of distribution you actually see in financial, healthcare, and HR data.

| Record type | Frequency |
|---|---|
| Routine | 60% |
| Review | 20% |
| Flagged | 10% |
| Incident | 6% |
| **Fraud** | **3%** |
| **Critical security** | **1%** |

Then we did what enterprise teams do: we simulated iterative retraining. Each generation,
a model trained on the current synthetic dataset generates the next round of synthetic
data. We ran this for 10 generations.

We tracked one key metric: **tail coverage entropy** — a measure of how well the synthetic
dataset preserves the diversity of its rare record types. An entropy of 1.0 means perfect
preservation. An entropy of 0.0 means those record types have completely disappeared.

---

## What We Found

| Generation | Tail Entropy | Fraud + Security Records | Status |
|---|---|---|---|
| 0 (original) | 0.854 (100%) | 4.6% of dataset | — |
| 1 | 0.775 (91%) | 3.4% | — |
| 2 | 0.722 (85%) | 2.9% | ⚠️ Warning threshold crossed |
| 3 | 0.650 (76%) | 2.3% | |
| 4 | 0.525 (61%) | 1.9% | |
| 5 | 0.391 (46%) | 1.75% | ❌ Critical threshold crossed |
| 7 | 0.352 (41%) | 1.3% | |
| 10 | 0.000 (0%) | 0.0% | ❌ Fraud records completely absent |

**By generation 5, the dataset has lost more than half of its tail record entropy.**
**By generation 10, fraud and critical security records are completely absent.**

A fraud detection model trained on generation-5 synthetic data will have near-zero
recall on fraud — not because of any bug in your code, not because of any change in
your data pipeline, but because the generative feedback loop has quietly erased the
very examples the model needs to learn from.

---

## Why This Happens

The mechanism is well-understood in the LLM literature [Shumailov et al., 2023],
but nobody had measured it in structured enterprise data before.

Each generation, rare records are slightly underrepresented in the synthetic output —
because generative models optimize for the majority of the distribution. By the next
generation, those slightly-underrepresented records become even rarer in the training
data. And so on.

It's a feedback loop with no natural stopping point.

The critical property: **this happens regardless of your differential privacy budget**.
You can set ε = 0.1 (maximum privacy) or ε = 10 (maximum utility) — the collapse
dynamics are driven by the generative feedback loop, not the privacy mechanism.
DP and model collapse are orthogonal failure modes.

Most enterprise teams have monitoring for the first one. Almost none monitor for the second.

---

## The Warning Signs You're Not Monitoring

Standard distribution statistics — mean, variance, KL divergence on the full distribution —
will not catch this early. They're dominated by the 80% of records that are fine.

Two metrics that do catch it:

**1. Tail coverage entropy**

Compute Shannon entropy only over the bottom 20% of value frequencies in your field of interest.
At generation 2 in our study, full-distribution entropy had dropped only 4%, but tail entropy
had already dropped 15%. The early warning is in the tail, not the mean.

```python
from model_collapse.metrics import tail_coverage_entropy, entropy_within_tolerance

tail_h = tail_coverage_entropy(synthetic_dataset, field="record_type")
if not entropy_within_tolerance(tail_h, original_tail_h, tolerance=0.10):
    raise AlertException("Model collapse warning: tail entropy dropped >10%")
```

**2. Minority class survival rate**

Track the fraction of records belonging to your high-stakes minority classes
(fraud, security incidents, rare diagnoses) at each generation.

In our dataset: fraud started at 3%, dropped to 2.3% by generation 3, and hit 0% by generation 10.

If your fraud rate in synthetic data is declining generation over generation, you have collapse.

---

## What Actually Works

We tested three mitigation strategies:

| Strategy | Gen 5 Tail Entropy | vs. Baseline | Verdict |
|---|---|---|---|
| **No mitigation** | 0.391 | −51% | ❌ Fails at gen 5 |
| **Real-data anchoring** (20% real records injected) | 0.662 | −18% | ❌ Insufficient at 30% collapse rate |
| **Diversity-rewarded sampling** | 0.986 | +22% | ✅ Works |
| **Combined (both)** | 0.971 | +20% | ✅ Works |

**Diversity-rewarded sampling** up-weights rare records inversely proportional to their frequency
before passing them to the next training iteration. Fraud records — which appear in 3% of the
dataset — get a sampling weight approximately 4× higher than routine records. This counteracts
the generative model's tendency to forget them.

The combined strategy (diversity sampling + real-data anchoring) is the most robust and is
what we recommend for production pipelines.

---

## How to Add This to Your Pipeline Right Now

The monitoring code is two functions in pure Python with no dependencies:

```python
# Install: pip install git+https://github.com/anote-ai/Research-EnterpriseSynth
from model_collapse.metrics import tail_coverage_entropy, minority_class_representation
from model_collapse.mitigation import mitigated_pipeline_step

# Before each synthetic data generation:
current_tail_h = tail_coverage_entropy(current_dataset, field="record_type")
minority_rep = minority_class_representation(
    current_dataset, field="record_type",
    minority_classes=["fraud", "critical_security"]
)

# Alert if degrading
if current_tail_h < 0.9 * original_tail_h:
    print(f"Warning: tail entropy at {current_tail_h:.3f} ({current_tail_h/original_tail_h:.1%} of original)")

# Apply mitigation before generating next round
next_generation_input = mitigated_pipeline_step(
    current_dataset,
    original_real_dataset,
    field="record_type",
    strategy="both",  # diversity sampling + real-data anchoring
)
```

---

## The Broader Implication

Model collapse in synthetic data is not a corner case. It affects any pipeline that:
- Uses a generative model (GAN, VAE, LLM) to produce training data
- Retrains on synthetic outputs more than 3–4 times
- Has a long-tail distribution with compliance-critical rare events

This covers: fraud detection, anomaly detection, clinical decision support, security
threat detection, and any enterprise ML system where rare events matter most.

The fix is not to stop using synthetic data. The fix is to monitor tail entropy and
apply diversity controls — two additions to your pipeline that take a few hours to implement.

---

## What's Next

This is one finding from the EnterpriseSynth benchmark. The full benchmark also covers:
- ε selection guidance for GDPR, HIPAA, and SOX contexts (with concrete utility retention numbers)
- DP noise mechanism comparison: Gaussian vs. Laplace vs. Discrete for enterprise tabular schemas
- PII detection rate as a privacy metric for synthetic documents
- Cross-vendor comparison of major synthetic data tools on enterprise schemas

The benchmark is open-source and all code is reproducible:
https://github.com/anote-ai/Research-EnterpriseSynth

Paper in preparation for NeurIPS 2026 Datasets & Benchmarks.

---

## References

Shumailov, I., Shumaylov, Z., Zhao, Y., Gal, Y., Papernot, N., & Anderson, R. (2023).
*The curse of recursion: Training on generated data makes models forget.*
arXiv:2305.17493.

Gerstgrasser, M. et al. (2024). *Is model collapse inevitable? Breaking the curse of
recursion by accumulating real and synthetic data.* arXiv:2404.01413.

EnterpriseSynth benchmark code: https://github.com/anote-ai/Research-EnterpriseSynth

---

*Rashmi Thimmaraju is a researcher at anote AI working on privacy-preserving synthetic
data for regulated enterprise applications. For benchmark participation inquiries,
reach out via the GitHub repository.*
