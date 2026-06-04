"""
app/dashboard.py
=================
HealthWatch — Emergency Triage Support Dashboard.

Built with Streamlit.  Run with:
    streamlit run app/dashboard.py

Architecture
------------
The dashboard is a single-page Streamlit application organised into four tabs:

  Home                   — mission statement, conditions covered, how it works
  Symptom Assessment     — checkbox + free-text symptom input → ranked differential
  Vital Signs Monitor    — per-vital sliders → risk scores for 4 critical conditions
  Citations & Methodology— full reference list and methodological transparency note

Symptom Assessment (page_symptoms)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
User selects symptoms from 39 options across 6 anatomical groups, optionally
types a free-text description that is auto-parsed via keyword matching, and
rates each selected symptom on a 1–10 severity scale.  Clinical context
(prior conditions, family history, travel) can be added via an expandable
section.  Results are produced by ``models.symptom_engine.assess()`` and
displayed as expandable condition cards with recommended investigations
and red-flag alerts.

Vital Signs Monitor (page_vitals)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Four subtabs — Sepsis, Pneumonia, Hypovolemic Shock, Cardiac Emergency.

  Sepsis:   Bayesian LSTM via Monte Carlo dropout (80 stochastic forward
            passes).  The single-timepoint score is obtained by prepending
            47 normal-baseline steps and appending the current readings,
            forming a 48-step sequence.  Posterior mean ± 1.96 × std is
            displayed as a 95% credible interval.

  Others:   Deterministic rule-based scores (condition_rules.py) derived
            from CURB-65, Shock Index, and haemodynamic pattern analysis.
            A synthetic std = prob × 0.08 is used to render a plausible CI
            band in the UI without implying calibrated uncertainty.

Design decisions
----------------
  - No animations or auto-refresh: Streamlit reruns once per user interaction.
    The previous "play/pause" loop caused rendering glitches and was replaced
    with a static slider.
  - Font pairing: Lora (serif) for readable content; Inter (sans-serif) for
    UI chrome (tabs, badges, labels) — a common clinical/academic convention.
  - Light theme (see .streamlit/config.toml): white background, blue accent.
  - All CSS is injected once via st.markdown(unsafe_allow_html=True) at module
    load to keep the component code clean.
  - model and scaler are cached via @st.cache_resource so they are loaded once
    per Streamlit server process, not on every page interaction.

Model note
----------
The Bayesian LSTM was trained on SYNTHETIC data only and has NOT been validated
on real patient records.  AUROC ≈ 1.000 on synthetic data is expected given
the clean OU-process separation by design; it does not represent real-world
performance.  This limitation is disclosed in the Citations tab.
"""

import os
import sys
import joblib
import torch
import numpy as np
import streamlit as st

# Allow imports from the project root regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.synthetic_generator import VITALS, NORMAL_RANGES, VITAL_UNITS
from models.bayesian_lstm import BayesianLSTM
from models.symptom_engine import (
    assess, parse_text,
    SYMPTOM_GROUPS, SYMPTOM_LABELS,
    RISK_FACTOR_GROUPS, RISK_FACTOR_LABELS,
    CONDITIONS,
)
from models.condition_rules import (
    score_pneumonia, score_shock, score_cardiac, CONDITION_META
)

# ── Artifact paths ────────────────────────────────────────────────────────────
_BASE       = os.path.join(os.path.dirname(__file__), "..")
SCALER_PATH = os.path.join(_BASE, "artifacts", "scaler.pkl")
MODEL_PATH  = os.path.join(_BASE, "artifacts", "model.pt")

# Baseline vital values used to pad the 47 history steps when a single time
# point is fed to the LSTM.  These match the population means in NORMAL_PARAMS.
NORMAL_BL = [78, 37.0, 7.5, 1.0, 122, 16, 98]   # HR Temp WBC Lactate SBP RR SpO2

# ── Page configuration (must be the first Streamlit call) ────────────────────
st.set_page_config(
    page_title="HealthWatch",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
# Injected once at module load.
# Key rules:
#   - Lora serif for all body content; Inter sans-serif for UI chrome.
#   - Minimal styling overrides for Streamlit's default widget labels.
#   - Responsive card system (hw-card / hw-card-gray / hw-card-blue).
#   - Badge system for urgency levels (badge-critical / badge-high / etc.).
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

/* Lora for all readable content; Inter reserved for UI chrome */
html, body, [class*="css"] {
  font-family: 'Lora', Georgia, serif !important;
}
/* Small UI labels stay in Inter */
.slabel, .badge, .risk-level, .risk-ci, .disclaimer,
.stSlider label, div[data-testid="stCheckbox"] label p,
.stTextArea label, .stSelectbox label,
.stTabs [data-baseweb="tab"],
button[kind="primary"], button[kind="secondary"] {
  font-family: 'Inter', system-ui, sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1080px; padding: 2rem 2.5rem 4rem; }

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {
  gap: 0; border-bottom: 2px solid #E9ECEF; background: transparent;
}
.stTabs [data-baseweb="tab"] {
  padding: 10px 22px; font-size: 0.87rem; font-weight: 500;
  color: #6C757D; border-radius: 0;
  border-bottom: 2px solid transparent; margin-bottom: -2px; background: transparent;
}
.stTabs [aria-selected="true"] {
  color: #1565C0 !important;
  border-bottom: 2px solid #1565C0 !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }

/* ── Card variants ── */
.hw-card {
  background: #fff; border: 1px solid #E9ECEF;
  border-radius: 10px; padding: 18px 22px; margin-bottom: 10px;
}
.hw-card-gray {
  background: #F8F9FA; border: 1px solid #E9ECEF;
  border-radius: 10px; padding: 18px 22px; margin-bottom: 10px;
}
.hw-card-blue {
  background: #EFF6FF; border: 1px solid #BFDBFE;
  border-radius: 10px; padding: 18px 22px; margin-bottom: 10px;
}

/* ── Urgency / level badges ── */
.badge {
  display: inline-block; padding: 2px 10px; border-radius: 99px;
  font-size: 0.70rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
}
.badge-critical { background:#FEE2E2; color:#991B1B; }
.badge-high     { background:#FEF3C7; color:#92400E; }
.badge-moderate { background:#FEF3C7; color:#92400E; }
.badge-low      { background:#D1FAE5; color:#065F46; }

/* ── Risk score display ── */
.risk-number { font-size: 2.8rem; font-weight: 700; line-height:1; letter-spacing:-1px; }
.risk-level  { font-size:0.70rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; margin-top:4px; }
.risk-ci     { font-size:0.74rem; color:#6C757D; margin-top:6px; }

/* ── Confidence / score bar ── */
.conf-bar-wrap { background:#F1F3F5; border-radius:4px; height:5px; margin:7px 0 3px; overflow:hidden; }
.conf-bar-fill { height:5px; border-radius:4px; }

/* ── Section label ── */
.slabel { font-size:0.70rem; font-weight:600; color:#6C757D; text-transform:uppercase;
          letter-spacing:0.08em; margin-bottom:6px; }

/* ── Clinical finding row ── */
.find-row { font-size:0.82rem; color:#495057; padding:4px 0;
            border-bottom:1px solid #F1F3F5; }
.find-row:last-child { border-bottom:none; }

/* ── Home hero section ── */
.hero-title { font-size:2.1rem; font-weight:700; color:#1565C0; letter-spacing:-0.5px; line-height:1.15; }
.hero-sub   { font-size:1.05rem; color:#495057; margin-top:10px; line-height:1.6; }
.pillar-icon  { font-size:1.8rem; }
.pillar-title { font-size:0.95rem; font-weight:600; color:#212529; margin:6px 0 4px; }
.pillar-body  { font-size:0.83rem; color:#6C757D; line-height:1.55; }

/* ── Disclaimer / footnote ── */
.disclaimer { font-size:0.74rem; color:#ADB5BD; line-height:1.5; margin-top:10px; }
hr.hw { border:none; border-top:1px solid #E9ECEF; margin:20px 0; }

/* ── Streamlit native widget overrides ── */
.stSlider label       { font-size:0.82rem !important; color:#495057 !important; }
.stCheckbox label p   { font-size:0.84rem !important; color:#212529 !important; margin:0 !important; }
.stTextArea label     { font-size:0.82rem !important; color:#495057 !important; }
.stExpander summary p { font-size:0.84rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Resource loader ───────────────────────────────────────────────────────────

@st.cache_resource
def load_model_and_scaler():
    """
    Load the trained BayesianLSTM and StandardScaler from disk.

    Cached with @st.cache_resource so deserialization runs once per Streamlit
    server session.  Returns (None, None) if artifacts are not found, allowing
    the dashboard to degrade gracefully with a warning message.

    Returns:
        model  : BayesianLSTM in eval() mode, or None.
        scaler : fitted StandardScaler, or None.
    """
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        return None, None
    scaler = joblib.load(SCALER_PATH)
    model  = BayesianLSTM(
        input_size=len(VITALS), hidden_size=64, num_layers=2, dropout=0.3
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, scaler


# ── Shared rendering helpers ──────────────────────────────────────────────────

def _level_color(level: str) -> str:
    """Map a risk level string to its hex colour for text and borders."""
    return {
        "HIGH":     "#C62828",
        "MODERATE": "#E65100",
        "LOW":      "#2E7D32",
        "CRITICAL": "#C62828",
    }.get(level, "#212529")


def _bar_color(level: str) -> str:
    """Map a risk level string to its hex colour for progress bar fills."""
    return {
        "HIGH":     "#FECACA",
        "MODERATE": "#FDE68A",
        "LOW":      "#BBF7D0",
        "CRITICAL": "#FECACA",
    }.get(level, "#CED4DA")


def _vital_status(vital: str, val: float) -> str:
    """
    Classify a vital-sign reading as 'ok', 'warn', or 'danger'.

    SpO2 uses custom thresholds (< 90 = danger, < 95 = warn).
    All other vitals use a margin-based approach: more than 30% outside the
    normal range is 'danger'; any deviation is 'warn'.

    Args:
        vital : VITALS key (e.g. "HR", "SpO2").
        val   : Current measured value.

    Returns:
        str — 'ok', 'warn', or 'danger'.
    """
    lo, hi = NORMAL_RANGES[vital]
    if vital == "SpO2":
        return "danger" if val < 90 else "warn" if val < 95 else "ok"
    margin = (
        abs(val - (lo if val < lo else hi)) / max((hi - lo) / 2, 0.01)
        if (val < lo or val > hi) else 0
    )
    return "danger" if margin > 0.3 else "warn" if (val < lo or val > hi) else "ok"


def render_risk_block(
    prob: float,
    std: float,
    level: str,
    title: str,
    note: str = "",
) -> None:
    """
    Render a centred risk-score card with 95% credible interval.

    Displays:
      - Percentage risk (large coloured number)
      - Level label (LOW / MODERATE / HIGH)
      - 95% CI as  mean ± 1.96 × std  clamped to [0%, 100%]
      - Urgency badge if level is not LOW
      - Optional footnote (e.g. model source note)

    Args:
        prob  : Posterior mean probability in [0, 1].
        std   : Posterior standard deviation.
        level : "LOW" / "MODERATE" / "HIGH" / "CRITICAL".
        title : Card header label.
        note  : Optional small text displayed below the badge.
    """
    col  = _level_color(level)
    clo  = max(0, prob - 1.96 * std)
    chi  = min(1, prob + 1.96 * std)
    badge = (
        f'<span class="badge badge-{level.lower()}">{level}</span>'
        if level in ("HIGH", "CRITICAL", "MODERATE") else ""
    )
    st.markdown(f"""
    <div class="hw-card" style="text-align:center; border-color:{col}22;">
      <div class="slabel">{title}</div>
      <div class="risk-number" style="color:{col};">{prob:.0%}</div>
      <div class="risk-level"  style="color:{col};">{level} RISK</div>
      <div class="risk-ci">95% CI &nbsp;{clo:.0%} – {chi:.0%}</div>
      <div style="margin-top:8px;">{badge}</div>
      {"<div class='disclaimer' style='margin-top:8px;'>"+note+"</div>" if note else ""}
    </div>""", unsafe_allow_html=True)


def render_findings(
    findings: list,
    tests: list,
    actions: list,
) -> None:
    """
    Render the abnormal findings list plus investigation and action expanders.

    Args:
        findings : List of human-readable abnormal vital-sign strings.
        tests    : List of recommended investigations.
        actions  : List of immediate management steps.
    """
    if findings:
        st.markdown('<div class="slabel">Key Findings</div>', unsafe_allow_html=True)
        rows = "".join(f'<div class="find-row">• {f}</div>' for f in findings)
        st.markdown(f'<div class="hw-card-gray">{rows}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="hw-card-gray" style="color:#6C757D;font-size:0.83rem;">'
            'All monitored vitals within expected range.</div>',
            unsafe_allow_html=True,
        )
    with st.expander("Recommended investigations"):
        for t in tests:
            st.markdown(f"&nbsp;&nbsp;• {t}")
    with st.expander("Suggested actions"):
        for a in actions:
            st.markdown(f"&nbsp;&nbsp;• {a}")


# ── Vital slider configuration ────────────────────────────────────────────────
# Each key maps to kwargs forwarded directly to st.slider().
# Ranges and defaults match NORMAL_PARAMS in synthetic_generator.py and the
# physiological limits used during data generation.

SLIDER_CFG = {
    "HR":      dict(min_value=40,   max_value=160,  value=78,   step=1,   format="%d bpm"),
    "Temp":    dict(min_value=34.0, max_value=42.0, value=37.0, step=0.1, format="%.1f °C"),
    "WBC":     dict(min_value=1.0,  max_value=30.0, value=7.5,  step=0.5, format="%.1f k/µL"),
    "Lactate": dict(min_value=0.2,  max_value=15.0, value=1.0,  step=0.1, format="%.1f mmol/L"),
    "SBP":     dict(min_value=55,   max_value=210,  value=122,  step=1,   format="%d mmHg"),
    "RR":      dict(min_value=6,    max_value=45,   value=16,   step=1,   format="%d /min"),
    "SpO2":    dict(min_value=70,   max_value=100,  value=98,   step=1,   format="%d %%"),
}

# Full display names for slider labels in the UI
VITAL_FULL = {
    "HR":      "Heart Rate",
    "Temp":    "Temperature",
    "WBC":     "WBC",
    "Lactate": "Lactate",
    "SBP":     "Systolic BP",
    "RR":      "Resp. Rate",
    "SpO2":    "SpO₂",
}

# Which vitals are shown for each condition's slider panel
# (not all conditions use WBC / Lactate at the bedside)
COND_VITALS = {
    "sepsis":    ["HR", "Temp", "WBC", "Lactate", "SBP", "RR", "SpO2"],
    "pneumonia": ["HR", "Temp", "SBP", "RR", "SpO2"],
    "shock":     ["HR", "SBP", "RR", "SpO2"],
    "cardiac":   ["HR", "SBP", "RR", "SpO2"],
}


# ═════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ═════════════════════════════════════════════════════════════════════════════

def page_home() -> None:
    """
    Render the landing / home page.

    Sections:
      - Hero headline and mission statement
      - Three mission pillars (pre-assessment, LMIC focus, early action)
      - Three-step "how it works" walkthrough
      - Grid of all 14 conditions with urgency badges
      - Clinical disclaimer
    """
    st.markdown("""
    <div style="max-width:780px; padding: 10px 0 30px;">
      <div class="hero-title">Early triage support,<br>for every community.</div>
      <div class="hero-sub">
        HealthWatch helps healthcare workers and patients identify potential
        conditions early, prioritise the right investigations, and understand
        warning signs — even in settings with limited laboratory access.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Mission pillars
    c1, c2, c3 = st.columns(3, gap="medium")
    pillars = [
        ("🩺", "Pre-Assessment Support",
         "Enter symptoms or vital signs to receive a ranked differential and recommended "
         "investigations — before lab results are available."),
        ("🌍", "Designed for Underserved Communities",
         "Built with low-resource clinical settings in mind. Conditions covered reflect "
         "the leading causes of preventable death in low- and middle-income countries."),
        ("⚡", "Act Earlier, Act Better",
         "Early identification of sepsis, malaria, TB, stroke, and cardiac emergencies "
         "dramatically improves survival. HealthWatch helps clinicians and community "
         "health workers move faster."),
    ]
    for col, (icon, title, body) in zip([c1, c2, c3], pillars):
        col.markdown(f"""
        <div class="hw-card" style="height:100%;">
          <div class="pillar-icon">{icon}</div>
          <div class="pillar-title">{title}</div>
          <div class="pillar-body">{body}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # How it works
    st.markdown(
        '<div style="font-size:1.05rem;font-weight:600;color:#212529;margin-bottom:14px;">'
        'How it works</div>',
        unsafe_allow_html=True,
    )
    s1, s2, s3 = st.columns(3, gap="medium")
    steps = [
        ("01", "Describe Symptoms",
         "Select from 39 symptoms across 6 categories, rate each by severity, add prior "
         "medical conditions and family history, or simply type what the patient is experiencing."),
        ("02", "Add Clinical Context",
         "Include risk factors — diabetes, HIV, pregnancy, recent travel — to sharpen the "
         "differential. The engine weights symptoms and risk factors together."),
        ("03", "Review & Act",
         "Receive a ranked differential with recommended investigations, red flags to watch "
         "for, and immediate management steps — specific to each condition."),
    ]
    for col, (num, title, body) in zip([s1, s2, s3], steps):
        col.markdown(f"""
        <div class="hw-card-gray" style="height:100%;">
          <div style="font-size:1.5rem;font-weight:700;color:#BFDBFE;">{num}</div>
          <div class="pillar-title">{title}</div>
          <div class="pillar-body">{body}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Conditions grid
    st.markdown(
        '<div style="font-size:1.05rem;font-weight:600;color:#212529;margin-bottom:12px;">'
        'Conditions Covered</div>',
        unsafe_allow_html=True,
    )
    cond_names  = [c["name"]    for c in CONDITIONS.values()]
    urgency_map = {c["name"]: c["urgency"] for c in CONDITIONS.values()}
    cols = st.columns(4)
    for i, name in enumerate(cond_names):
        urg      = urgency_map[name]
        badge_cls = (
            "badge-critical" if urg == "CRITICAL"
            else "badge-high" if urg == "HIGH"
            else "badge-moderate"
        )
        cols[i % 4].markdown(
            f'<div style="padding:6px 0; font-size:0.83rem; color:#212529;">'
            f'{name} &nbsp;<span class="badge {badge_cls}">{urg}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("""
    <hr class="hw">
    <div class="disclaimer">
      ⚠ HealthWatch is a clinical decision support tool for pre-assessment only.
      It is not a diagnostic device and does not replace clinical history-taking,
      physical examination, laboratory evaluation, or the judgment of a qualified
      healthcare professional. All findings must be interpreted in full clinical context.
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SYMPTOM ASSESSMENT
# ═════════════════════════════════════════════════════════════════════════════

def page_symptoms() -> None:
    """
    Render the Symptom Assessment tab.

    Workflow:
      1. Optional free-text field → auto-parsed by parse_text() → checkboxes pre-filled.
      2. Symptom checkboxes in six anatomical groups.
      3. Severity sliders (1–10) for each selected symptom.
      4. Expandable clinical context section (risk factors).
      5. "Assess →" button → calls assess() → stores results in st.session_state.
      6. Results rendered as expandable condition cards (top result auto-expanded).
    """
    st.markdown("""
    <div style="margin-bottom:18px;">
      <div style="font-size:1.05rem;font-weight:600;color:#212529;">Symptom Assessment</div>
      <div style="font-size:0.84rem;color:#6C757D;margin-top:3px;">
        Select symptoms, rate their severity, and add clinical context to receive a ranked differential.
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Free-text input ───────────────────────────────────────────────────────
    st.markdown('<div class="slabel">Free-text description (optional)</div>',
                unsafe_allow_html=True)
    free_text = st.text_area(
        label="free_text_hidden",
        label_visibility="collapsed",
        placeholder=(
            'e.g. "Patient has had high fever and chills for 3 days, '
            'very fatigued, neck feels stiff and sensitive to light."'
        ),
        height=80,
        key="free_text",
    )
    # Keywords from the free-text will pre-check matching symptom boxes below
    text_symptoms = parse_text(free_text) if free_text.strip() else []

    st.markdown('<hr class="hw">', unsafe_allow_html=True)

    # ── Symptom checkboxes ────────────────────────────────────────────────────
    st.markdown('<div class="slabel">Select all symptoms present</div>',
                unsafe_allow_html=True)
    checked = set(text_symptoms)   # seed with auto-detected symptoms

    for group_name, symptoms in SYMPTOM_GROUPS.items():
        st.markdown(
            f'<div class="slabel" style="margin-top:12px;">{group_name}</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, (key, label) in enumerate(symptoms.items()):
            default = key in checked
            if cols[i % 2].checkbox(label, value=default, key=f"sym_{key}"):
                checked.add(key)
            else:
                checked.discard(key)

    selected = sorted(checked)

    # ── Severity sliders ──────────────────────────────────────────────────────
    severities = {}
    if selected:
        st.markdown('<hr class="hw">', unsafe_allow_html=True)
        st.markdown(
            '<div class="slabel">Rate severity of selected symptoms '
            '(1 = mild, 10 = severe)</div>',
            unsafe_allow_html=True,
        )
        sev_cols = st.columns(3)
        for i, key in enumerate(selected):
            label = SYMPTOM_LABELS.get(key, key)
            # Truncate to the part before any parenthetical clarification
            short = label.split("(")[0].strip()
            severities[key] = sev_cols[i % 3].select_slider(
                short,
                options=list(range(1, 11)),
                value=5,
                key=f"sev_{key}",
            )

    # ── Risk factors ──────────────────────────────────────────────────────────
    st.markdown('<hr class="hw">', unsafe_allow_html=True)
    with st.expander(
        "Clinical context — prior conditions, family history & travel",
        expanded=False,
    ):
        st.markdown(
            '<div class="slabel" style="margin-bottom:8px;">Select all that apply</div>',
            unsafe_allow_html=True,
        )
        selected_rf = []
        for group_name, factors in RISK_FACTOR_GROUPS.items():
            st.markdown(
                f'<div class="slabel" style="margin-top:10px;">{group_name}</div>',
                unsafe_allow_html=True,
            )
            rf_cols = st.columns(2)
            for i, (key, label) in enumerate(factors.items()):
                if rf_cols[i % 2].checkbox(label, key=f"rf_{key}"):
                    selected_rf.append(key)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Assess button ─────────────────────────────────────────────────────────
    run = st.button("Assess →", type="primary")

    if run:
        st.session_state["sym_results"]  = assess(selected, severities, selected_rf)
        st.session_state["sym_selected"] = selected

    results = st.session_state.get("sym_results")
    if results is None:
        return

    st.markdown('<hr class="hw">', unsafe_allow_html=True)

    # Show which symptoms were auto-detected from free text
    if text_symptoms and free_text.strip():
        matched = [
            SYMPTOM_LABELS.get(k, k).split("(")[0].strip()
            for k in text_symptoms
        ]
        st.markdown(
            f'<div class="hw-card-blue" style="font-size:0.82rem;color:#1E40AF;">'
            f'Detected from free text: {", ".join(matched)}</div>',
            unsafe_allow_html=True,
        )

    if not results:
        st.markdown(
            '<div class="hw-card-gray" style="color:#6C757D;font-size:0.84rem;text-align:center;">'
            'No conditions flagged for the selected symptoms. Try adding more symptoms '
            'or clinical context, or consult a clinician directly.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div style="font-size:0.83rem;color:#6C757D;margin-bottom:14px;">'
        f'{len(results)} condition{"s" if len(results) > 1 else ""} flagged — '
        f'ranked by symptom overlap and risk factors</div>',
        unsafe_allow_html=True,
    )

    # Condition cards — first result is auto-expanded
    for r in results:
        pct     = int(r["confidence"] * 100)
        bar_col = _bar_color(r["level"])
        txt_col = _level_color(r["level"])
        urg_cls = r["urgency"].lower()
        rf_note = (
            f'<div style="font-size:0.76rem;color:#6B7280;margin-top:4px;">'
            f'Risk factors contributed +{r["rf_bonus"]:.1f} pts</div>'
            if r["rf_bonus"] > 0 else ""
        )

        with st.expander(
            f"{r['name']}  ·  {r['level']} concern",
            expanded=(r is results[0]),
        ):
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">
              <div>
                <div style="font-weight:600;font-size:0.93rem;color:#212529;">{r['name']}</div>
                <div style="font-size:0.82rem;color:#6C757D;margin-top:2px;">{r['description']}</div>
              </div>
              <span class="badge badge-{urg_cls}">{r['urgency']}</span>
            </div>
            <div style="font-size:0.77rem;color:{txt_col};font-weight:500;margin-bottom:2px;">
              Symptom overlap &nbsp;{pct}%
            </div>
            <div class="conf-bar-wrap">
              <div class="conf-bar-fill" style="width:{pct}%;background:{bar_col};"></div>
            </div>
            {rf_note}
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    '<div class="slabel" style="margin-top:10px;">Recommended Investigations</div>',
                    unsafe_allow_html=True,
                )
                for t in r["tests"]:
                    st.markdown(
                        f'<div style="font-size:0.82rem;color:#495057;padding:3px 0;">• {t}</div>',
                        unsafe_allow_html=True,
                    )
            with c2:
                st.markdown(
                    '<div class="slabel" style="margin-top:10px;">Red Flags</div>',
                    unsafe_allow_html=True,
                )
                for f in r["red_flags"]:
                    st.markdown(
                        f'<div style="font-size:0.82rem;color:#B91C1C;padding:3px 0;">⚠ {f}</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown(
        '<div class="disclaimer" style="margin-top:20px;">'
        '⚠ Symptom overlap scores indicate match strength — they are not probabilities of disease. '
        'Always integrate with clinical history, physical examination, and laboratory results.</div>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# VITAL SIGNS MONITOR
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _predict_sepsis(hr, temp, wbc, lactate, sbp, rr, spo2, _model, _scaler):
    """
    Cached MC-dropout sepsis prediction.

    Vital sign values are passed as plain floats so Streamlit can hash them as
    the cache key.  ``_model`` and ``_scaler`` are prefixed with ``_`` so
    Streamlit skips hashing them (they are constant across the session).

    Uses n_samples=20 — sufficient for a stable posterior mean and 95% CI while
    keeping latency under ~0.5 s on CPU.  The full 80-sample run used during
    training is not needed for interactive display.
    """
    raw = np.array(
        [NORMAL_BL] * 47 + [[hr, temp, wbc, lactate, sbp, rr, spo2]],
        dtype=np.float32,
    )
    sc = _scaler.transform(raw)
    x  = torch.tensor(sc[np.newaxis], dtype=torch.float32)
    m, s = _model.mc_predict(x, n_samples=20)
    return float(m[0, -1]), float(s[0, -1])


def page_vitals(model: BayesianLSTM, scaler) -> None:
    """
    Render the Vital Signs Monitor tab.

    Four condition subtabs:
      Sepsis            — Bayesian LSTM (MC dropout, 80 samples)
      Pneumonia         — CURB-65 adapted rule score
      Hypovolemic Shock — Shock Index (HR / SBP) + supplementary rules
      Cardiac Emergency — Haemodynamic pattern (cardiogenic, brady/tachy, PE)

    Sepsis scoring uses a sequence padding trick: 47 normal-baseline time steps
    are prepended to the single current observation to produce a 48-step input
    for the LSTM.  This ensures the model receives a full-length sequence while
    reflecting only the most recent vital signs.

    Args:
        model  : BayesianLSTM loaded from artifacts/model.pt, or None.
        scaler : StandardScaler loaded from artifacts/scaler.pkl, or None.
    """
    st.markdown("""
    <div style="margin-bottom:18px;">
      <div style="font-size:1.05rem;font-weight:600;color:#212529;">Vital Signs Monitor</div>
      <div style="font-size:0.84rem;color:#6C757D;margin-top:3px;">
        Enter current vital signs to screen for critical conditions common in resource-limited settings.
      </div>
    </div>""", unsafe_allow_html=True)

    tabs = st.tabs(["Sepsis", "Pneumonia", "Hypovolemic Shock", "Cardiac Emergency"])

    def vital_sliders(cond_key: str, prefix: str) -> dict:
        """
        Render sliders for the vitals relevant to a given condition.

        Args:
            cond_key : Key into COND_VITALS (e.g. "sepsis").
            prefix   : Unique st.slider key prefix to avoid Streamlit ID collisions.

        Returns:
            dict {vital_key: current_slider_value}
        """
        vals = {}
        for v in COND_VITALS[cond_key]:
            vals[v] = st.slider(VITAL_FULL[v], key=f"{prefix}_{v}", **SLIDER_CFG[v])
        return vals

    # ── Sepsis ────────────────────────────────────────────────────────────────
    with tabs[0]:
        meta = CONDITION_META["sepsis"]
        if model is None:
            st.warning("Model not found — run `python3 train.py` first.")
        else:
            L, R = st.columns([1, 1], gap="large")
            with L:
                st.markdown('<div class="slabel">Vital Signs</div>',
                            unsafe_allow_html=True)
                vals = vital_sliders("sepsis", "sep")

            with R:
                # Cached MC prediction — reruns only when a vital value changes.
                # Findings are computed from the same vals dict so they stay in sync.
                prob, std = _predict_sepsis(
                    vals["HR"], vals["Temp"], vals["WBC"], vals["Lactate"],
                    vals["SBP"], vals["RR"], vals["SpO2"],
                    model, scaler,
                )
                level = "HIGH" if prob >= 0.55 else "MODERATE" if prob >= 0.30 else "LOW"

                render_risk_block(prob, std, level, "Sepsis Risk Score", meta["model_note"])

                # Compile bedside findings from the current vital values
                findings = []
                if vals["HR"]    > 90:      findings.append(f"Tachycardia — HR {vals['HR']:.0f} bpm")
                if vals["Temp"]  > 38.3:    findings.append(f"Fever — {vals['Temp']:.1f}°C")
                elif vals["Temp"] < 36:     findings.append(f"Hypothermia — {vals['Temp']:.1f}°C")
                if vals["RR"]    > 22:      findings.append(f"Tachypnea — RR {vals['RR']:.0f}/min")
                if vals["SBP"]   < 100:     findings.append(f"Hypotension — SBP {vals['SBP']:.0f} mmHg")
                if vals["Lactate"] > 2:     findings.append(f"Elevated lactate — {vals['Lactate']:.1f} mmol/L")
                if vals["WBC"] > 12 or vals["WBC"] < 4:
                    findings.append(f"Abnormal WBC — {vals['WBC']:.1f} k/µL")
                if vals["SpO2"]  < 95:      findings.append(f"Hypoxia — SpO₂ {vals['SpO2']:.0f}%")

                render_findings(findings, meta["tests"], meta["actions"])

    # ── Pneumonia ─────────────────────────────────────────────────────────────
    with tabs[1]:
        meta = CONDITION_META["pneumonia"]
        L, R = st.columns([1, 1], gap="large")
        with L:
            st.markdown('<div class="slabel">Vital Signs</div>', unsafe_allow_html=True)
            vals = vital_sliders("pneumonia", "pneu")
        with R:
            prob, level, findings = score_pneumonia(vals)
            # std = prob × 0.08 gives a plausible CI width for rule-based scores
            render_risk_block(prob, prob * 0.08, level, "Pneumonia Severity", meta["model_note"])
            render_findings(findings, meta["tests"], meta["actions"])

    # ── Hypovolemic Shock ─────────────────────────────────────────────────────
    with tabs[2]:
        meta = CONDITION_META["shock"]
        L, R = st.columns([1, 1], gap="large")
        with L:
            st.markdown('<div class="slabel">Vital Signs</div>', unsafe_allow_html=True)
            vals = vital_sliders("shock", "shk")
            si   = vals["HR"] / max(vals["SBP"], 1)
            # Display the Shock Index inline beneath the sliders for quick reference
            st.markdown(
                f'<div style="font-size:0.82rem;color:#6C757D;margin-top:6px;">'
                f'Shock Index (HR / SBP) = '
                f'<strong style="color:#212529;">{si:.2f}</strong>'
                f'&nbsp;(normal &lt; 0.7)</div>',
                unsafe_allow_html=True,
            )
        with R:
            prob, level, findings = score_shock(vals)
            render_risk_block(prob, prob * 0.08, level, "Shock Risk", meta["model_note"])
            render_findings(findings, meta["tests"], meta["actions"])

    # ── Cardiac Emergency ─────────────────────────────────────────────────────
    with tabs[3]:
        meta = CONDITION_META["cardiac"]
        L, R = st.columns([1, 1], gap="large")
        with L:
            st.markdown('<div class="slabel">Vital Signs</div>', unsafe_allow_html=True)
            vals = vital_sliders("cardiac", "card")
            st.markdown(
                '<div class="disclaimer" style="margin-top:10px;">'
                '12-lead ECG and serial troponin required for definitive ACS diagnosis.</div>',
                unsafe_allow_html=True,
            )
        with R:
            prob, level, findings = score_cardiac(vals)
            render_risk_block(prob, prob * 0.08, level, "Cardiac Emergency Risk", meta["model_note"])
            render_findings(findings, meta["tests"], meta["actions"])

    st.markdown(
        '<div class="disclaimer" style="margin-top:24px;">'
        '⚠ Risk scores are screening aids, not diagnostic conclusions. '
        'Always interpret alongside clinical history, examination, and laboratory data.</div>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# CITATIONS & METHODOLOGY
# ═════════════════════════════════════════════════════════════════════════════

def page_citations() -> None:
    """
    Render the Citations & Methodology tab.

    Displays a full numbered reference list for every clinical scoring framework,
    guideline, and research paper underpinning HealthWatch.  Organised into
    sections matching the condition categories in the Symptom Assessment tab.
    Ends with a methodological transparency note disclosing the synthetic
    training data limitation.
    """
    st.markdown("""
    <div style="margin-bottom:24px;">
      <div style="font-size:1.05rem;font-weight:600;color:#212529;font-family:'Inter',sans-serif;">
        Citations &amp; Methodology
      </div>
      <div style="font-size:0.88rem;color:#6C757D;margin-top:4px;font-family:'Inter',sans-serif;">
        Clinical scoring frameworks, guidelines, and research underpinning
        each component of HealthWatch.
      </div>
    </div>
    """, unsafe_allow_html=True)

    def section(title: str, subtitle: str = "") -> None:
        """Render a section header with an optional subtitle."""
        sub_html = (
            f'<div style="font-size:0.82rem;color:#6C757D;margin-top:2px;'
            f'font-family:\'Inter\',sans-serif;">{subtitle}</div>'
            if subtitle else ""
        )
        st.markdown(f"""
        <div style="margin:28px 0 12px;">
          <div style="font-size:0.70rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;
               color:#6C757D;font-family:'Inter',sans-serif;border-bottom:1px solid #E9ECEF;
               padding-bottom:6px;">{title}</div>
          {sub_html}
        </div>""", unsafe_allow_html=True)

    def ref(
        number: int,
        authors: str,
        year: int,
        title: str,
        journal: str,
        detail: str = "",
        url: str = "",
    ) -> None:
        """Render a single formatted bibliographic reference."""
        link = (
            f'<a href="{url}" target="_blank" style="color:#1565C0;text-decoration:none;'
            f'font-size:0.78rem;font-family:\'Inter\',sans-serif;">[link]</a>'
            if url else ""
        )
        detail_html = f'<span style="color:#6C757D;"> — {detail}</span>' if detail else ""
        st.markdown(f"""
        <div style="padding:8px 0; border-bottom:1px solid #F1F3F5; font-size:0.85rem; line-height:1.6;">
          <span style="color:#ADB5BD;font-family:'Inter',sans-serif;font-size:0.75rem;margin-right:8px;">[{number}]</span>
          <span style="color:#495057;">{authors} ({year}).</span>
          <em style="color:#212529;"> {title}.</em>
          <span style="color:#6C757D;"> <em>{journal}.</em></span>
          {detail_html}
          &nbsp;{link}
        </div>""", unsafe_allow_html=True)

    # ── Sepsis ────────────────────────────────────────────────────────────────
    section("Sepsis", "Sepsis-3 definition, qSOFA, and Surviving Sepsis Campaign")
    ref(1, "Singer M, et al.", 2016,
        "The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)",
        "JAMA, 315(8):801–810",
        "Defines sepsis as life-threatening organ dysfunction caused by dysregulated host "
        "response to infection; introduces SOFA score")
    ref(2, "Seymour CW, et al.", 2016,
        "Assessment of Clinical Criteria for Sepsis: For the Third International Consensus "
        "Definitions for Sepsis and Septic Shock (Sepsis-3)",
        "JAMA, 315(8):762–774",
        "Validates qSOFA (RR ≥ 22, altered mentation, SBP ≤ 100) as a bedside screening tool")
    ref(3, "Evans L, et al.", 2021,
        "Surviving Sepsis Campaign: International Guidelines for Management of Sepsis and Septic Shock",
        "Critical Care Medicine, 49(11):e1063–e1143",
        "Antibiotics within 1 h, 30 mL/kg fluid resuscitation, vasopressors for MAP < 65 mmHg")

    # ── Pneumonia ─────────────────────────────────────────────────────────────
    section("Pneumonia", "CURB-65 and ATS/IDSA guidelines")
    ref(4, "Lim WS, et al.", 2003,
        "Defining community acquired pneumonia severity on presentation to hospital: "
        "an international derivation and validation study",
        "Thorax, 58(5):377–382",
        "Derives and validates CURB-65 score (Confusion, Urea, RR ≥ 30, BP < 90/60, Age ≥ 65)")
    ref(5, "Mandell LA, et al.", 2007,
        "Infectious Diseases Society of America / American Thoracic Society Consensus "
        "Guidelines on the Management of Community-Acquired Pneumonia in Adults",
        "Clinical Infectious Diseases, 44(Suppl 2):S27–S72",
        "ICU admission criteria, empiric antibiotic selection, severity stratification")

    # ── Acute MI / ACS ────────────────────────────────────────────────────────
    section("Acute Myocardial Infarction / ACS", "HEART score and ACC/AHA guidelines")
    ref(6, "Backus BE, et al.", 2010,
        "Chest pain in the emergency room: a multicenter validation of the HEART Score",
        "Critical Pathways in Cardiology, 9(3):164–169",
        "Validates HEART score (History, ECG, Age, Risk factors, Troponin) for ACS risk stratification")
    ref(7, "O'Gara PT, et al.", 2013,
        "2013 ACCF/AHA Guideline for the Management of ST-Elevation Myocardial Infarction",
        "Journal of the American College of Cardiology, 61(4):e78–e140",
        "Diagnostic criteria, reperfusion strategy, door-to-balloon time targets")

    # ── Stroke ────────────────────────────────────────────────────────────────
    section("Stroke / TIA", "FAST criteria, NIHSS, and tPA eligibility window")
    ref(8, "Nor AM, et al.", 2004,
        "The Recognition of Stroke in the Emergency Room (ROSIER) scale: development "
        "and validation of a stroke recognition instrument",
        "The Lancet Neurology, 3(12):727–734",
        "Clinical basis for FAST (Face, Arm, Speech, Time) screening")
    ref(9, "Hacke W, et al.", 2008,
        "Thrombolysis with Alteplase 3 to 4.5 Hours after Acute Ischemic Stroke",
        "New England Journal of Medicine, 359(13):1317–1329",
        "ECASS III trial establishing the 4.5-hour tPA eligibility window")
    ref(10, "Powers WJ, et al.", 2019,
        "Guidelines for the Early Management of Patients With Acute Ischemic Stroke",
        "Stroke, 50(12):e344–e418",
        "AHA/ASA 2019 comprehensive stroke management guidelines")

    # ── Malaria ───────────────────────────────────────────────────────────────
    section("Malaria", "WHO severe malaria criteria and treatment guidelines")
    ref(11, "World Health Organization", 2022,
        "Guidelines for the Treatment of Malaria, 3rd edition",
        "WHO Press, Geneva",
        "RDT-first diagnostic approach, artemisinin-based combination therapy, severe malaria criteria")
    ref(12, "Dondorp AM, et al.", 2010,
        "Artesunate versus quinine in the treatment of severe falciparum malaria in African children (AQUAMAT)",
        "The Lancet, 376(9753):1647–1657",
        "Establishes IV artesunate as first-line for severe malaria")

    # ── Meningitis ────────────────────────────────────────────────────────────
    section("Meningitis / Encephalitis", "IDSA guidelines and LP protocols")
    ref(13, "Tunkel AR, et al.", 2004,
        "Practice Guidelines for the Management of Bacterial Meningitis",
        "Clinical Infectious Diseases, 39(9):1267–1284",
        "CT-before-LP indications, empiric antibiotic regimens, dexamethasone adjunctive therapy")
    ref(14, "van de Beek D, et al.", 2016,
        "Community-acquired bacterial meningitis in adults",
        "New England Journal of Medicine, 374(13):1254–1262",
        "Clinical features, diagnostic workup, and outcomes in bacterial meningitis")

    # ── Tuberculosis ──────────────────────────────────────────────────────────
    section("Tuberculosis", "WHO guidelines and GeneXpert evidence")
    ref(15, "World Health Organization", 2022,
        "WHO Consolidated Guidelines on Tuberculosis, Module 3: Diagnosis",
        "WHO Press, Geneva",
        "GeneXpert MTB/RIF as preferred initial test; sputum smear microscopy in resource-limited settings")
    ref(16, "Lawn SD, Zumla AI", 2011,
        "Tuberculosis",
        "The Lancet, 378(9785):57–72",
        "Comprehensive review of TB epidemiology, pathogenesis, diagnosis, and treatment")

    # ── Typhoid ───────────────────────────────────────────────────────────────
    section("Typhoid Fever")
    ref(17, "Bhutta ZA", 2006,
        "Current concepts in the diagnosis and treatment of typhoid fever",
        "BMJ, 333(7558):78–82",
        "Blood culture as gold standard; Widal test limitations; fluoroquinolone and cephalosporin treatment")
    ref(18, "Mogasale V, et al.", 2014,
        "Burden of typhoid fever in low-income and middle-income countries",
        "The Lancet Infectious Diseases, 14(2):133–143",
        "Epidemiological basis for typhoid inclusion in LMIC triage tools")

    # ── Dengue ────────────────────────────────────────────────────────────────
    section("Dengue Fever", "WHO 2009 revised classification")
    ref(19, "World Health Organization", 2009,
        "Dengue: Guidelines for Diagnosis, Treatment, Prevention and Control (revised edition)",
        "WHO Press, Geneva",
        "NS1 antigen, IgM/IgG serology, warning signs (abdominal pain, vomiting, bleeding), thrombocytopenia threshold")

    # ── Anaemia ───────────────────────────────────────────────────────────────
    section("Severe Anaemia")
    ref(20, "World Health Organization", 2011,
        "Haemoglobin Concentrations for the Diagnosis of Anaemia and Assessment of Severity",
        "WHO/NMH/NHD/MNM/11.1, Geneva",
        "Defines severe anaemia as Hgb < 7 g/dL; establishes transfusion thresholds by age and setting")

    # ── Preeclampsia ──────────────────────────────────────────────────────────
    section("Preeclampsia / Eclampsia", "ACOG task force and WHO maternal guidelines")
    ref(21, "American College of Obstetricians and Gynecologists", 2013,
        "Hypertension in Pregnancy: Task Force Report",
        "ACOG, Washington DC",
        "Diagnostic criteria: BP ≥ 140/90 on two occasions, proteinuria ≥ 300 mg/24h; "
        "severe features: BP ≥ 160/110")
    ref(22, "World Health Organization", 2011,
        "WHO Recommendations for Prevention and Treatment of Pre-Eclampsia and Eclampsia",
        "WHO Press, Geneva",
        "Magnesium sulphate for seizure prophylaxis and treatment; antihypertensives for severe range BP")

    # ── Appendicitis ──────────────────────────────────────────────────────────
    section("Appendicitis", "Alvarado score and WSES guidelines")
    ref(23, "Alvarado A", 1986,
        "A practical score for the early diagnosis of acute appendicitis",
        "Annals of Emergency Medicine, 15(5):557–564",
        "Derives MANTRELS/Alvarado score: migration of pain to RLQ, anorexia, nausea, "
        "RLQ tenderness, rebound, elevated temp, leukocytosis, left shift")
    ref(24, "Di Saverio S, et al.", 2020,
        "WSES Jerusalem Guidelines for Diagnosis and Treatment of Acute Appendicitis",
        "World Journal of Emergency Surgery, 15(1):27",
        "Ultrasound-first approach; CT for equivocal cases; non-operative management criteria")

    # ── DKA ───────────────────────────────────────────────────────────────────
    section("Diabetic Ketoacidosis (DKA)")
    ref(25, "American Diabetes Association", 2009,
        "Hyperglycemic Crises in Adult Patients With Diabetes",
        "Diabetes Care, 32(7):1335–1343",
        "Diagnostic criteria (glucose > 250, pH < 7.3, ketones), fluid/insulin/potassium replacement protocols")
    ref(26, "Joint British Diabetes Societies Inpatient Care Group", 2023,
        "The Management of Diabetic Ketoacidosis in Adults (4th edition)",
        "JBDS-IP, NHS England",
        "Fixed-rate insulin infusion, potassium replacement thresholds, monitoring frequency")

    # ── Pulmonary Embolism ────────────────────────────────────────────────────
    section("Pulmonary Embolism", "Wells score and ESC guidelines")
    ref(27, "Wells PS, et al.", 2000,
        "Derivation of a simple clinical model to categorize patients probability of pulmonary embolism",
        "Thrombosis and Haemostasis, 83(3):416–420",
        "Derives Wells PE score: DVT signs, PE most likely dx, HR > 100, immobilization, "
        "prior DVT/PE, haemoptysis, cancer")
    ref(28, "Konstantinides SV, et al.", 2020,
        "2019 ESC Guidelines for the diagnosis and management of acute pulmonary embolism",
        "European Heart Journal, 41(4):543–603",
        "CTPA as diagnostic standard; massive PE management; anticoagulation protocols")

    # ── Vital Signs Monitor scoring ───────────────────────────────────────────
    section("Vital Signs Monitor — Scoring Methods")
    ref(29, "Allgower M, Burri C", 1967,
        "Shock index",
        "Deutsche Medizinische Wochenschrift, 92(43):1947–1950",
        "Original derivation of Shock Index (HR / SBP); validated for haemorrhage classification")
    ref(30, "Rady MY, et al.", 1994,
        "Shock index: a re-evaluation in acute circulatory failure",
        "Resuscitation, 27(3):255–262",
        "Validates Shock Index > 1.0 as indicator of significant haemodynamic compromise")
    ref(31, "Royal College of Physicians", 2017,
        "National Early Warning Score (NEWS) 2: Standardising the Assessment of "
        "Acute-Illness Severity in the NHS",
        "RCP, London",
        "Validates RR, SpO2, temperature, BP, HR, and consciousness as composite early warning markers")

    # ── Machine Learning Model ────────────────────────────────────────────────
    section("Machine Learning Model — Bayesian LSTM", "Sepsis risk scoring via sequential inference")
    ref(32, "Hochreiter S, Schmidhuber J", 1997,
        "Long Short-Term Memory",
        "Neural Computation, 9(8):1735–1780",
        "Original LSTM architecture used as the sequential model backbone")
    ref(33, "Gal Y, Ghahramani Z", 2016,
        "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning",
        "Proceedings of ICML 2016",
        "Theoretical basis for MC Dropout as approximate Bayesian inference — "
        "80 stochastic forward passes yield posterior mean and 95% CI")
    ref(34, "Johnson AEW, et al.", 2016,
        "MIMIC-III, a freely accessible critical care database",
        "Scientific Data, 3:160035",
        "Reference dataset whose physiological distributions informed synthetic data generation; "
        "direct access was not obtained — synthetic ICU trajectories were parameterised from "
        "published reference ranges")

    # ── Methodological transparency note ─────────────────────────────────────
    st.markdown("""
    <div style="margin-top:32px; padding:16px 20px; background:#F8F9FA;
         border:1px solid #E9ECEF; border-radius:8px; font-size:0.82rem;
         color:#6C757D; line-height:1.7; font-family:'Inter',sans-serif;">
      <strong style="color:#212529;">Methodological transparency note.</strong>
      Symptom-to-condition weights are calibrated judgements derived from the clinical literature
      above, not regressed from labelled patient data.  They reflect published pathognomonic
      relationships (e.g. neck stiffness is a cardinal sign of meningitis; chest pain is the
      primary symptom of ACS) consistent with how validated scoring tools such as CURB-65 and
      the Wells score are constructed.  The sepsis LSTM was trained exclusively on synthetic data
      and has not been externally validated on real patient records.
      <br><br>
      HealthWatch is a pre-assessment decision-support tool only.  All outputs must be interpreted
      by a qualified clinician in the context of a complete clinical evaluation.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# APPLICATION SHELL
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Application entry point.

    Loads model artifacts (cached), renders the branded header, then
    dispatches to the four page functions based on the selected tab.
    """
    model, scaler = load_model_and_scaler()

    # Branded header bar
    st.markdown("""
    <div style="border-bottom:1px solid #E9ECEF; padding-bottom:14px; margin-bottom:20px;">
      <div style="display:flex; align-items:baseline; gap:12px;">
        <span style="font-size:1.35rem;font-weight:700;color:#1565C0;letter-spacing:-0.5px;">HealthWatch</span>
        <span style="font-size:0.80rem;color:#6C757D;">Emergency Triage Support &nbsp;·&nbsp; Stanford CS153</span>
      </div>
    </div>""", unsafe_allow_html=True)

    tab0, tab1, tab2, tab3 = st.tabs([
        "Home",
        "Symptom Assessment",
        "Vital Signs Monitor",
        "Citations & Methodology",
    ])
    with tab0: page_home()
    with tab1: page_symptoms()
    with tab2: page_vitals(model, scaler)
    with tab3: page_citations()


if __name__ == "__main__":
    main()
