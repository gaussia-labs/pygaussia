# Chatbot PII Evaluation Sandbox (v3)

## Purpose

This sandbox answers a single practical question: **which PII detection model should we use in production before a chatbot pipeline stores, analyses, or forwards user conversations?**

When a chatbot processes real conversations — customer support, banking, healthcare — user messages contain sensitive data: names, IBANs, credit cards, SSNs, and more. Before that data flows downstream, a model must detect and mask it. The wrong choice means leaking PII at scale.

The sandbox takes 500 real-format chatbot conversations with ground-truth PII annotations, feeds the same conversations (without annotations) to each candidate model, and measures how well each one finds the sensitive entities. The scoring formula deliberately penalises **false negatives** more than false positives — in privacy, missing an IBAN or SSN is far more damaging than flagging something extra.

The five OpenMed models are clinical NER fine-tunes with vocabulary similar to support/healthcare chatbots. They are benchmarked against Microsoft Presidio (a regex-based baseline) to determine whether neural models provide meaningful uplift in this domain.

The output is a ranked table — "Presidio scores 35/100, the best OpenMed reaches 62/100 → use that model with these mitigations" — that gives an engineering team a documented, reproducible basis for their tooling decision before going to production.

---

Experimental sandbox for benchmarking Named Entity Recognition (NER) and pattern-matching models on their ability to detect **Personally Identifiable Information (PII)** within chatbot conversations.

The evaluation produces a **Domain-Adjusted Privacy Detection Score (v3)** — a composite score that penalises models for missed critical entities, incomplete domain coverage, and regulatory misalignment. Models are evaluated in their **native state**: no custom regex/NER hybrid layers are applied on top.

**Changes from v2:**
- REGEX and NER supplementation layers removed — models run as-is.
- InfraScore (latency/cost) removed — focus is exclusively on detection capability.
- Score formula reduced from 6 components to 5.

---

## File Structure

```
tests/privacy/
├── Analisis_Score_Chatbot_v3.py          # Main evaluation script
├── panel_chatbot_v3.py                   # HTML dashboard generator
├── chatbot_conversations_500.txt         # Untagged corpus — model input (500 turns)
├── chatbot_conversations_tagged_500.txt  # Tagged ground-truth corpus (500 turns)
├── eval_untagged_100.txt                 # Untagged subset for fast runs (100 turns)
├── eval_tagged_100.txt                   # Tagged subset for fast runs (100 turns)
├── chatbot_v3_results.json               # Serialised evaluation output
└── chatbot_v3_results_panel_v3.html      # Interactive HTML dashboard
```

---

## Corpus Format

Both corpus files share the same line-by-line structure. The **tagged** file uses inline XML-style annotations; the **untagged** file is the same content with tags stripped. Lines starting with `---` and blank lines are ignored.

**Tagged format (ground truth):**
```
CHATBOT: Hello! Welcome to QuickSupport.
USER: My name is <PERSON>Kimberly Elliott</PERSON> and my email is <EMAIL_ADDRESS>k@example.com</EMAIL_ADDRESS>.
USER: My IBAN is <IBAN_CODE>GB21FTNS15634939958769</IBAN_CODE>.
```

**Untagged format (model input):**
```
CHATBOT: Hello! Welcome to QuickSupport.
USER: My name is Kimberly Elliott and my email is k@example.com.
USER: My IBAN is GB21FTNS15634939958769.
```

Supported speaker prefixes (stripped before evaluation): `CHATBOT`, `PATIENT`, `CUSTOMER`, `USER`, `EMPLOYEE`.

---

## PII Taxonomy

Nine canonical entity classes derived from the tagged corpus. Both criticality weights (`w`) and false-negative severity weights (`rho`) are symmetric in this sandbox version and must sum exactly to `1.0`.

| Class | Description | Weight (`w` and `rho`) |
|:---|:---|---:|
| `person` | Full or partial individual names | 0.15 |
| `email_address` | Electronic mail addresses | 0.15 |
| `iban_code` | Bank account/routing numbers (IBAN) | 0.15 |
| `credit_card` | Credit card numbers | 0.13 |
| `phone_number` | Telephone contact details | 0.12 |
| `ip_address` | IP addresses and device identifiers | 0.09 |
| `street_address` | Physical addresses and locations | 0.08 |
| `us_ssn` | US Social Security Numbers | 0.08 |
| `date_time` | Temporal context and dates | 0.05 |
| **Total** | | **1.00** |

### Label Aliasing

Models use their own label vocabularies. Before evaluation, all predicted labels are normalised through `canonicalize()`:

1. BIO/BILOU prefixes are stripped (`B-`, `I-`, `E-`, `S-`, `U-`, `L-`).
2. The label is lowercased and spaces/hyphens replaced with underscores.
3. `LABEL_ALIASES` remaps model-specific labels to the canonical taxonomy.

Key aliases include: `first_name` / `last_name` → `person`; `date` / `date_of_birth` → `date_time`; `bank_routing_number` / `account_number` → `iban_code`; `mac_address` → `ip_address`. Predictions with labels outside the 9-class taxonomy are silently discarded — they do not inflate false-positive counts.

---

## Score Formula (v3)

```
Score(M, d) = DetectionScore × Coverage × DomainFit × RegulatoryFit × Penalty_FN
```

### Components

**DetectionScore** — weighted F₂ across all canonical classes:
```
DetectionScore(M, d) = Σ_c [ w(c,d) × F2(M, c) ]
```
F₂ is used because it weights Recall twice as heavily as Precision. In PII detection, a missed entity (false negative) is costlier than a spurious one (false positive).

**Coverage** — proportion of canonical classes the model can detect in principle:
```
Coverage(M, d) = |Supported(M) ∩ Taxonomy| / |Taxonomy|
```
For HuggingFace models this is read from `model.config.id2label`. For Presidio it is derived from its standard recogniser set.

**DomainFit** — fixed coefficient indicating training alignment with the target domain (default: `0.90`).

**RegulatoryFit** — fixed coefficient indicating alignment with GDPR/HIPAA requirements (default: `0.85`).

**Penalty_FN** — penalty for critical omissions:
```
CriticalFN(M, d) = Σ_c [ rho(c,d) × FNrate(M, c) ]
Penalty_FN(M, d) = max(0, 1 − CriticalFN)
```
A model that consistently misses high-severity classes (e.g., `iban_code`, `credit_card`) will have a high `CriticalFN` and a low `Penalty_FN`, significantly depressing its final score.

### Score Interpretation

| Range | Label |
|:---|:---|
| < 40 | No adecuado |
| 40 – 59 | Adecuado solo como línea base o complemento |
| 60 – 74 | Adecuado con mitigaciones |
| 75 – 84 | Operacionalmente adecuado |
| 85 – 94 | Recomendado |
| ≥ 95 | Recomendado con fuerte evidencia local |

---

## Risk Profile Indices

Three risk indices are computed per model:

**R1 — Weakest class risk** (concentrated vulnerability):
```
R1(M, d) = max_c [ rho(c,d) × FNrate(M, c) ]
```

**R2 — Systemic risk** (distributed coverage and omission failures):
```
R2(M, d) = 1 − Penalty_FN(M, d) × Coverage(M, d)
```

**R_final — Combined risk** (probabilistic union):
```
R_final(M, d) = 1 − (1 − R1) × (1 − R2)
```
`R_final` is only `0` when both `R1` and `R2` are zero.

---

## Models Evaluated

### HuggingFace NER (requires `transformers` + `torch`)

| Name | HuggingFace path |
|:---|:---|
| `OpenMed-PII-SuperClinical-434M` | `openmed/OpenMed-PII-SuperClinical-Large-434M-v1` |
| `OpenMed-PII-BigMed-BioClinical` | `openmed/OpenMed-PII-BigMed-Large-278M-v1` |
| `OpenMed-PII-ModernMed-Large-395M` | `OpenMed/OpenMed-PII-ModernMed-Large-395M-v1` |
| `OpenMed-PII-SuperMedical-Large-355M` | `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1` |
| `OpenMed-privacy-filter-nemotron` | `OpenMed/privacy-filter-nemotron` |

All five are clinical NER fine-tunes. They are loaded via `transformers.pipeline("ner", aggregation_strategy="simple")`. If `transformers` is not installed and `SKIP_HF_IF_UNAVAILABLE = True`, these models are recorded as failed with `score_100 = 0.0` and the script continues.

### Presidio Baseline

| Name | Paradigm |
|:---|:---|
| `Presidio-Baseline` | `presidio_baseline` |

Microsoft Presidio with a blank spaCy model (`spacy.blank("en")`) — no neural network download required. Uses standard regex-based recognisers. Supports: `person`, `email_address`, `phone_number`, `iban_code`, `credit_card`, `date_time`, `ip_address`, `us_ssn`, `street_address`.

---

## Span Matching

Predictions are matched to ground-truth spans using IoU (Intersection over Union). A prediction counts as a True Positive when:

1. Its label matches the ground-truth label exactly (after canonicalisation).
2. Its IoU with the ground-truth span is ≥ `IOU_THRESHOLD` (default: `0.50`).

Overlapping predictions for the same span are resolved greedily: the highest-confidence prediction is kept, others are discarded.

---

## Configuration

All parameters are at the top of `Analisis_Score_Chatbot_v3.py`:

| Parameter | Default | Description |
|:---|:---|:---|
| `DEVICE` | `-1` | HuggingFace inference device. `-1` = CPU, `0` = first GPU. |
| `IOU_THRESHOLD` | `0.50` | Minimum IoU for a prediction to count as TP. |
| `USE_PARALLEL` | `False` | Run HuggingFace models in parallel (disabled to avoid CPU memory saturation). |
| `MAX_WORKERS` | `2` | Thread count when `USE_PARALLEL = True`. |
| `SAVE_RESULTS_JSON` | `True` | Whether to write the output JSON file. |
| `OUTPUT_JSON_PATH` | `chatbot_v3_results.json` | Output path for the JSON results. |
| `SKIP_HF_IF_UNAVAILABLE` | `True` | Skip HuggingFace models silently if `transformers` is not installed. |
| `TAGGED_FILE` | `chatbot_conversations_tagged_500.txt` | Ground-truth corpus. |
| `UNTAGGED_FILE` | `chatbot_conversations_500.txt` | Model-input corpus. |

To run on the 100-turn subset instead, set:
```python
TAGGED_FILE   = "eval_tagged_100.txt"
UNTAGGED_FILE = "eval_untagged_100.txt"
```

---

## Quick Start

### 1. Install dependencies

`presidio-analyzer` and `spacy` are auto-installed on first run. To also evaluate HuggingFace models:

```bash
pip install transformers torch
```

### 2. Run the evaluation

```bash
python Analisis_Score_Chatbot_v3.py
```

Output is saved to `chatbot_v3_results.json`.

### 3. Generate the HTML dashboard

```bash
python panel_chatbot_v3.py chatbot_v3_results.json --open
```

Options:
- `-o <path>` — custom output path for the HTML file (default: `<input>_panel_v3.html`).
- `--open` — open the dashboard in the default browser after generation.

---

## Output Format

### `chatbot_v3_results.json`

A JSON array, one object per model, sorted by `score_100` descending. Key fields:

| Field | Type | Description |
|:---|:---|:---|
| `name` | string | Model name |
| `paradigm` | string | `"huggingface"` or `"presidio_baseline"` |
| `success` | bool | Whether evaluation completed without error |
| `score_100` | float | Final score (0–100) |
| `interpretation` | string | Qualitative label |
| `detection_score` | float | Weighted F₂ component |
| `coverage` | float | Fraction of taxonomy classes supported |
| `domain_fit` | float | Domain alignment coefficient |
| `regulatory_fit` | float | Regulatory alignment coefficient |
| `penalty_fn` | float | False-negative penalty (0–1) |
| `critical_fn` | float | Raw critical FN score |
| `r1_weakest_class_risk_100` | float | R1 as percentage |
| `r2_systemic_risk_100` | float | R2 as percentage |
| `r_final_100` | float | Combined risk as percentage |
| `evaluation.metrics_by_label` | object | Per-class TP/FP/FN/precision/recall/F₂/FN-rate |
| `evaluation.micro_metrics` | object | Micro-aggregated TP/FP/FN/precision/recall/F₂ |

### `*_panel_v3.html`

Self-contained HTML dashboard (requires internet for Chart.js CDN). Features:
- Best model highlight card
- Risk strip with `R_final` per model
- Sortable ranking table with score bars
- Per-model detail panel: score components, risk indices, micro metrics, per-class F₂ and FN Rate chart
- Scatter plot: Score vs. R_final
- Cross-model F₂ comparison chart
- Cross-model FN Rate comparison chart
