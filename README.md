# HealthWatch

### Emergency triage support for every community.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Stanford CS153](https://img.shields.io/badge/Stanford-CS153-8C1515?style=flat)

> Built by **Liv Weiner** · Stanford CS153: Frontier Systems · 2025

HealthWatch helps healthcare workers, community health workers, and first responders pre-assess potential diagnoses from symptoms and bedside vital signs — before any laboratory result is available. Built with low-resource clinical settings in mind, it covers the leading causes of preventable death in low- and middle-income countries (LMICs).

---

```
┌─────────────────────────────────────────────────────────────────────────┐
│  HealthWatch              Emergency Triage Support · Stanford CS153     │
├──────────────┬──────────────────────┬──────────────────────────────────┤
│     Home     │  Symptom Assessment  │  Vital Signs Monitor  │ Citations│
└──────────────┴──────────────────────┴──────────────────────────────────┘

        Early triage support,
        for every community.

  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
  │ 🩺 Pre-Assessment│  │ 🌍 Built for     │  │ ⚡ Act Earlier,  │
  │    Support       │  │    Underserved   │  │    Act Better    │
  │                  │  │    Communities   │  │                  │
  │ Enter symptoms   │  │ Conditions cover │  │ Early ID of      │
  │ or vitals to get │  │ leading LMIC     │  │ sepsis, TB,      │
  │ a ranked         │  │ causes of death  │  │ stroke, malaria  │
  │ differential     │  │                  │  │ saves lives      │
  └──────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## Two ways to assess

### 1 · Symptom Assessment

Type a free-text description or check symptoms from 39 options across 6 categories. The keyword parser handles both lay language ("throwing up") and clinical terms ("emesis") — matched symptoms pre-fill automatically.

Rate each symptom **1–10 for severity**. Mild counts half, severe counts 1.5×. Add prior conditions, family history, or travel history to activate the risk factor bonus layer.

```
 Symptom Assessment
 ─────────────────────────────────────────────────────

 Free-text description (optional)
 ┌──────────────────────────────────────────────────┐
 │ Patient has high fever, neck stiffness, and is   │
 │ sensitive to light. Very confused.               │
 └──────────────────────────────────────────────────┘
 Detected from free text: Fever, Neck stiffness, Photophobia, Confusion

 ─────────────────────────────────────────────────────
 3 conditions flagged — ranked by symptom overlap and risk factors

 ▼ Meningitis / Encephalitis  ·  HIGH concern              [CRITICAL]
   ┌──────────────────────────────────────────────────────┐
   │ Symptom overlap  100%                                │
   │ ██████████████████████████████████████████████████  │
   │                                                      │
   │  Investigations              Red Flags               │
   │  • CT head before LP         ⚠ Non-blanching rash    │
   │  • Lumbar puncture           ⚠ Declining GCS         │
   │  • Blood cultures ×2         ⚠ Seizures              │
   └──────────────────────────────────────────────────────┘

 ▶ Sepsis  ·  MODERATE concern                             [CRITICAL]
 ▶ Typhoid Fever  ·  LOW concern                           [HIGH]
```

**14 conditions scored in parallel:**

| Urgency | Conditions |
|---------|------------|
| Critical | Sepsis · Acute MI / ACS · Stroke / TIA · Meningitis · Preeclampsia · DKA · Pulmonary Embolism |
| High | Pneumonia · Malaria · Tuberculosis · Typhoid Fever · Dengue Fever · Appendicitis · Severe Anaemia |

Risk factors that sharpen the differential: 21 options across prior medical conditions, family history, and travel/exposure history.

---

### 2 · Vital Signs Monitor

Enter current bedside readings on sliders. Each condition has its own subtab with a slider panel on the left and a live-updating risk card on the right.

```
 Vital Signs Monitor
 ──────────────────────────────────────────────────────────────────────

 [ Sepsis ] [ Pneumonia ] [ Hypovolemic Shock ] [ Cardiac Emergency ]

  Vital Signs                        ┌────────────────────────────────┐
  ───────────────────                │       SEPSIS RISK SCORE        │
  Heart Rate  ────●────── 118 bpm    │                                │
  Temperature ───────●─── 38.8 °C   │             73%                │
  WBC         ──────────●─ 14 k/µL  │          HIGH RISK             │
  Lactate     ────────●── 3.2 mmol  │      95% CI   48% – 98%        │
  Systolic BP ●──────────  88 mmHg  │                                │
  Resp. Rate  ──────●──── 24 /min   │            [HIGH]              │
  SpO₂        ─────────●── 94%      │  Bayesian LSTM — trained on    │
                                    │  synthetic ICU data            │
                                    └────────────────────────────────┘

  Key Findings
  ┌──────────────────────────────────────────────┐
  │ • Tachycardia — HR 118 bpm                   │
  │ • Fever — 38.8 °C                            │
  │ • Hypotension — SBP 88 mmHg                 │
  │ • Elevated lactate — 3.2 mmol/L             │
  │ • Abnormal WBC — 14.0 k/µL                 │
  └──────────────────────────────────────────────┘
  ▶ Recommended investigations
  ▶ Suggested actions
```

| Condition | Vitals used | Method |
|-----------|-------------|--------|
| **Sepsis** | HR, Temp, WBC, Lactate, SBP, RR, SpO₂ | Bayesian LSTM — MC Dropout |
| **Pneumonia** | HR, Temp, SBP, RR, SpO₂ | CURB-65 adapted — Lim et al., *Thorax* 2003 |
| **Hypovolemic Shock** | HR, SBP, RR, SpO₂ | Shock Index — Allgower & Burri, *DMW* 1967 |
| **Cardiac Emergency** | HR, SBP, RR, SpO₂ | Haemodynamic pattern — O'Gara et al., *JACC* 2013 |

---

### Citations & Methodology

The app's fourth tab renders **34 numbered references** — each with authors, year, journal, and a one-line note on exactly what it supports in HealthWatch. A transparency note at the bottom discloses the synthetic training data limitation and explains that symptom weights are expert-calibrated, not statistically regressed.

---

## How it works

**Core rule: every number is computed, never generated.**

All scores come from deterministic formulas or trained model weights. No LLM runs at query time. Claude was used during development — not at runtime.

### Scoring pipeline

```
User input: symptoms + severities + risk factors
     │
     ▼
 symptom_engine.assess()
     ├─ base_score  = Σ weight(condition, symptom) × severity_multiplier(1–10)
     │               severity:  1–3 → 0.5×  │  4–6 → 1.0×  │  7–10 → 1.5×
     ├─ rf_bonus    = Σ RISK_BONUSES[risk_factor][condition]
     ├─ confidence  = min(1.0, total / condition_threshold)
     └─ level:  ≥ 0.60 HIGH  │  ≥ 0.35 MODERATE  │  ≥ 0.12 LOW  │  < 0.12 suppressed

User input: vital signs
     │
     ├─► Sepsis  →  BayesianLSTM.mc_predict(x, n_samples=80)
     │       ├─ prepend 47 normal-baseline steps + 1 current → 48-step sequence
     │       ├─ 80 stochastic forward passes (dropout active in train() mode)
     │       └─ posterior mean ± 1.96 × std  →  risk % + 95% credible interval
     │
     └─► Pneumonia / Shock / Cardiac  →  condition_rules.score_*(vitals)
             └─ deterministic formula  →  (probability, level, findings list)
```

### Model training pipeline

```
generate_dataset()
  600 patients × 48 hours  ·  35% sepsis prevalence
  Ornstein-Uhlenbeck process per vital  ·  sigmoid drift to sepsis targets post-onset
          │
          ▼
  StandardScaler  (fit on train split only — no leakage into validation)
          │
          ▼
  BayesianLSTM
  2 LSTM layers  ·  hidden=64  ·  dropout=0.3
  BCE loss  ·  Adam lr=1e-3  ·  ReduceLROnPlateau  ·  early stopping patience=12
  gradient clip max-norm 1.0
          │
          ▼
  MC inference  ·  50 samples per patient  ·  batched in groups of 64
          │
          ▼
  evaluate_all()
  AUROC  ·  PR curve  ·  sensitivity / specificity  ·  lead-time vs. qSOFA baseline
          │
          ▼
  artifacts/  →  model.pt  ·  scaler.pkl  ·  artifacts.pkl
```

### 5 scoring methods

| Method | Input | Output |
|--------|-------|--------|
| Symptom engine | Symptom keys + severities + risk factors | Confidence ratio per condition (0–1) |
| CURB-65 adapted | RR, SBP, SpO₂, Temp, HR | Normalised pneumonia severity |
| Shock Index | HR, SBP, RR, SpO₂ | Shock probability with SI inline |
| Haemodynamic pattern | HR, SBP, RR, SpO₂ | Cardiac emergency probability |
| Bayesian LSTM | 48-step vital-sign sequence | Posterior mean risk + 95% CI |

---

## Tech stack

| Layer | Choice |
|-------|--------|
| UI | Streamlit ≥ 1.30 · Python |
| ML | PyTorch · 2-layer LSTM · MC Dropout |
| Analytics | NumPy · Pandas · scikit-learn |
| Storage | Joblib (model.pt, scaler.pkl, artifacts.pkl on disk) |
| Typography | Lora serif (content) · Inter sans-serif (UI chrome) |
| Clinical evidence | 34 peer-reviewed sources — WHO · ACC/AHA · ESC · IDSA · Lancet · NEJM · JAMA |

No LangChain · No vector DB · No Docker · No API keys at runtime

---

## Getting started

```bash
# Clone
git clone https://github.com/Liv-Weiner/CS153-Bayesian-Reasoning-Sepsis-Detection.git
cd CS153-Bayesian-Reasoning-Sepsis-Detection/sepsis_detection

# Install
pip install -r requirements.txt

# Train (once — ~2–5 min on CPU)
python train.py

# Run
streamlit run app/dashboard.py
# Opens at http://localhost:8501
```

---

## Project structure

```
sepsis_detection/
├── train.py                    # Full pipeline: data → model → evaluation → artifacts
├── requirements.txt
├── data/
│   └── synthetic_generator.py # OU simulator — 600 patients × 48 h
├── models/
│   ├── bayesian_lstm.py        # 2-layer LSTM + MC Dropout
│   ├── condition_rules.py      # CURB-65, Shock Index, cardiac haemodynamic rules
│   └── symptom_engine.py       # 14 conditions · 39 symptoms · 21 risk factors
├── evaluation/
│   └── metrics.py              # AUROC, PR curve, sensitivity/specificity, lead-time
├── app/
│   └── dashboard.py            # Streamlit app — 4 tabs
└── artifacts/                  # Generated by train.py
    ├── model.pt
    ├── scaler.pkl
    └── artifacts.pkl
```

---

## Limitations

| # | Limitation |
|---|------------|
| 1 | **Synthetic training data.** LSTM trained on OU-process data only. AUROC ≈ 1.000 is a property of synthetic separation — not a real-world performance claim. |
| 2 | **Vitals-only.** Real sepsis diagnosis requires blood cultures, procalcitonin, and serial lactate. This tool assists pre-assessment before labs return. |
| 3 | **Expert-calibrated weights.** Symptom weights reflect published pathognomonic relationships but have not been validated against a labelled diagnosis dataset. |
| 4 | **Adapted, not exact, rule scores.** CURB-65 omits serum urea and age; SpO₂ is added as a bedside proxy. |

---

## AI Assistance

Claude (Anthropic) assisted with code scaffolding, documentation, and debugging throughout development.

---

## Disclaimer

HealthWatch is a pre-assessment decision-support tool only. It is not a diagnostic device and does not replace clinical evaluation by a qualified healthcare professional. In an emergency, call for emergency medical services immediately.

---

*Stanford CS153: Frontier Systems · 2025*
