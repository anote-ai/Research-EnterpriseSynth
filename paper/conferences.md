# EnterpriseSynth: Conference & Venue Targeting Strategy

**Paper:** *EnterpriseSynth: A Privacy-Utility-Fidelity Benchmark for Synthetic Data Generation in Regulated Enterprise Settings*

---

## 1. Primary Academic Targets

### Tier 1 — Highest Impact

| Venue | Track | ~Deadline | Notification | Why EnterpriseSynth Fits |
|---|---|---|---|---|
| **NeurIPS 2026 Datasets & Benchmarks** | D&B Track | May 2026 | Sep 2026 | Benchmark paper track; privacy + synthetic data among most active NeurIPS areas; D&B explicitly solicits evaluation frameworks |
| **ICML 2026** | Main / DP Workshop | Jan 2026 | Apr 2026 | Differential privacy is a core ICML topic; TSTR/bootstrap CI methodology aligns with ML community norms |
| **VLDB 2027** | Research Track | Mar 2027 | Jun 2027 | Enterprise data management is VLDB's core scope; schema corpus, constraint violation rate, and Pareto analysis are DB contributions |

### Tier 2 — Strong Fit

| Venue | Track | ~Deadline | Notification | Why EnterpriseSynth Fits |
|---|---|---|---|---|
| **TMLR** (Transactions on ML Research) | Rolling | Rolling | ~3 months | No deadline pressure; good for comprehensive benchmarks requiring reproducibility artifacts; open access |
| **ICLR 2027** | Main | Sep 2026 | Jan 2027 | Growing DP + fairness track; model collapse finding is relevant to ICLR's practical-impact audience |
| **KDD 2026** | Applied Data Science | Feb 2026 | May 2026 | Applied enterprise focus; industry practitioners attend; direct overlap with compliance use cases |

### Recommended First Submission: TMLR (rolling) or NeurIPS 2026 D&B

TMLR allows iterative revision and reproducibility review, which suits a benchmark paper with an accompanying code artifact. NeurIPS D&B has the highest visibility if the paper is strong enough for acceptance without major revisions.

---

## 2. Privacy-Specific Workshops & Venues

These venues reach the audience most likely to cite and extend EnterpriseSynth.

| Venue | Type | Co-located With | Submission Window | Priority |
|---|---|---|---|---|
| **TPDP** (Theory and Practice of Differential Privacy) | Workshop | NeurIPS / ICML (alternating) | ~8 weeks before parent conference | **Highest** — direct DP audience, workshop abstract only |
| **SyntheticData4ML** Workshop | Workshop | NeurIPS/ICML | ~8 weeks before parent | High — synthetic data focus |
| **IEEE S&P** (Security & Privacy) | Conference | Standalone | Nov / Mar cycles | Medium — if MIA evaluation section is expanded |
| **CCS** (ACM Computer & Communications Security) | Conference | Standalone | Jan / May cycles | Medium — strong MIA / privacy attack community |
| **PETS** (Privacy Enhancing Technologies) | Journal + Symposium | Standalone (Jul) | Rolling (PoPETs) | High — strong DP + enterprise privacy community; lower bar than IEEE S&P |

**Action**: Submit a 2-page extended abstract to TPDP workshop at the next NeurIPS/ICML cycle while the full paper is under review. This seeds community awareness with zero risk of scoop.

---

## 3. Industry & Compliance Community Venues

These are non-academic but high-value for enterprise adoption and partnership.

| Venue | Audience | Format | Relevance |
|---|---|---|---|
| **IAPP Global Privacy Summit** (Washington DC, Apr) | Privacy professionals, GCs, compliance officers | 30-min talk or poster | ε-selection decision guide is directly actionable for DPOs |
| **HIMSS Annual Conference** (Mar) | Healthcare IT, clinical informatics, EHR vendors | Workshop / demo | HIPAA synthetic data is top unmet need; EHR asset type directly applies |
| **American Bankers Association TechConnect** (Oct) | Financial services technology leaders | Presentation | SOX/financial synthetic data tier maps to ABA member use cases |
| **Strata Data & AI** (O'Reilly) | Data engineers, ML practitioners | Tutorial | Model collapse monitoring checklist is practitioner-ready content |
| **IAPP Europe Data Protection Congress** (Brussels, Nov) | GDPR compliance, European DPOs | Talk | GDPR ε-tier guidance is immediately usable by European DPOs |

**Priority**: HIMSS and IAPP Summit offer the clearest path to enterprise partnership; speaking slots create inbound interest from healthcare and financial synthetic data teams.

---

## 4. Partnership & Outreach Strategy

### 4.1 Synthetic Data Vendors (Benchmark Inclusion)

Including major vendors in the benchmark positions EnterpriseSynth as independent, authoritative, and comprehensive—the vendors benefit from third-party validation.

| Company | Contact Strategy | What We Offer | What We Ask |
|---|---|---|---|
| **Gretel.ai** | OSS community / LinkedIn outreach | Inclusion in benchmark results; co-authorship credit if they provide data | Run their APIs against our schema corpus; share held-out results |
| **Mostly AI** | Conference connection (NeurIPS/ICLR) | Same as above | Same as above |
| **Syntheticus** | Direct email | Benchmark credibility signal for enterprise sales | API access + results sharing |
| **Synthesized.io** | LinkedIn outreach | Independent evaluation for regulated industries | Run against HR schema corpus |
| **YData** | OSS community | Benchmark publication credit | API access |

**Template outreach message (adapt per company):**

> We're finalizing EnterpriseSynth, the first privacy-utility-fidelity benchmark for synthetic data in regulated enterprise settings (GDPR, HIPAA, SOX). We'd like to include [Company] in our evaluation so practitioners have a complete comparison. Participation means running your API against our enterprise schema corpus (~500 records, six asset types) and sharing the privacy/utility/fidelity scores. You'd be acknowledged in the paper and have first access to results. Would a 30-min call this month work?

### 4.2 Enterprise Data Science Research Partnerships

| Organization | Contact | Relevance |
|---|---|---|
| **JPMorgan AI Research** | Published researchers at NeurIPS/ICML | Financial transaction logs (SOX tier) direct use case |
| **Goldman Sachs Data Science** | Conference networking | Same as JPMorgan; internal synthetic data for model training |
| **Epic Systems** | HIMSS conference contact | EHR synthetic data; HIPAA compliance is immediate need |
| **Cerner (Oracle Health)** | HIMSS conference contact | Same as Epic; large existing synthetic data programs |
| **NHS Digital (UK)** | PETS / IAPP Europe | GDPR + clinical data; strict tier (ε ≤ 0.5) directly applies |
| **Microsoft Research (DP team)** | ICML/NeurIPS | DP noise mechanisms; potential co-author for statistical rigor section |

### 4.3 Academic Collaborators

| Name / Group | Institution | Connection Point |
|---|---|---|
| Aaron Roth | UPenn | Differential privacy theory; could review ε-tier mapping |
| Ilya Mironov | Google DeepMind | DP-SGD accountant; epsilon verification section |
| Mihaela van der Schaar group | Cambridge | Synthetic data for healthcare; TSTR methodology |
| Nicolas Papernot | Google Brain / UofT | Model collapse + DP intersection |

---

## 5. Submission Calendar

Current date: June 2026. Deadlines in chronological order:

| Deadline | Venue | Status | Priority Action |
|---|---|---|---|
| **Jun–Jul 2026** | TPDP @ NeurIPS 2026 workshop | Check exact date on TPDP website | Submit 2-page abstract now |
| **~Aug 2026** | NeurIPS 2026 D&B | Check CFP (D&B sometimes separate deadline) | Submit if paper is near-final |
| **Oct 2026** | IAPP Global Privacy Summit (proposal) | Speaking proposal window | Submit ε-selection talk proposal |
| **~Nov 2026** | TMLR (rolling) | Submit after NeurIPS decision | Primary fallback if NeurIPS rejected |
| **~Jan 2027** | ICML 2027 | Next ICML cycle | Full conference submission |
| **~Mar 2027** | VLDB 2027 | Research track | DB community submission |

---

## 6. Abstract Variants by Venue

Different venues need different framing of the same paper.

### ML community (NeurIPS / ICML)
> We present EnterpriseSynth, a benchmark for evaluating differential privacy mechanisms on enterprise synthetic data across six regulated data asset types. We quantify the privacy-utility-fidelity Pareto frontier at six ε values (0.1–10), introduce statistically rigorous reporting with bootstrap CIs and Wilcoxon signed-rank tests, and present the first multi-generation model collapse study for structured tabular data with mitigation strategies that maintain tail diversity within 10% over five retraining iterations.

### Database community (VLDB / SIGMOD)
> Enterprises in regulated industries lack principled tools to evaluate synthetic data generators against domain-specific schema constraints. EnterpriseSynth provides an enterprise schema corpus across six data asset types with formal inter-column constraint verification, a compliance-tier ε mapping for GDPR/HIPAA/SOX, and a model collapse detection framework targeting long-tail record preservation—critical for fraud and security incident records that appear at 1–3% base rates.

### Privacy community (PETS / TPDP)
> We characterize the empirical privacy-utility tradeoff for synthetic data in regulated enterprise settings across six differential privacy budgets, using membership inference AUC as the privacy metric. Key findings: (1) the balanced tier (ε=1–2) dominates the Pareto frontier for most enterprise use cases; (2) tighter DP budgets (ε≤0.5) introduce 3–5× training variance requiring multi-seed reporting; (3) iterative synthetic retraining without mitigation collapses tail-record diversity by 51% in five generations.

### Industry / compliance (IAPP / HIMSS)
> How do you choose the right privacy budget for your synthetic data program—and how do you know when your synthetic data is drifting away from the real thing? EnterpriseSynth answers both questions with a concrete decision framework: a three-tier ε selection guide aligned to GDPR, HIPAA, and SOX requirements, and a monitoring system that detects model collapse in iterative retraining before it affects fraud detection and compliance reporting.

---

## 7. Immediate Action Items

- [ ] **Check TPDP 2026 abstract deadline** — co-located with NeurIPS; submit 2-page abstract as soon as paper draft is stable
- [ ] **Check NeurIPS 2026 D&B track CFP** — confirm whether D&B has same abstract deadline as main track (~May 2026)
- [ ] **Gretel.ai outreach** — send benchmark inclusion email (template in Section 4.1)
- [ ] **Mostly AI outreach** — same template
- [ ] **HIMSS 2027 speaking proposal** — submit once benchmark results include EHR asset type
- [ ] **IAPP Privacy Summit 2027 proposal** — submit ε-selection decision guide talk (Oct–Nov 2026 proposal window)
- [ ] **TMLR pre-submission inquiry** — TMLR allows pre-submission inquiries to confirm scope fit before full submission
