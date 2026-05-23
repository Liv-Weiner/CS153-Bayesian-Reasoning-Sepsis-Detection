"""
SepsisWatch — Early Sepsis Detection Dashboard
Run from the project root:
    streamlit run app/dashboard.py
"""

import os
import sys
import joblib
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.synthetic_generator import VITALS, NORMAL_RANGES, VITAL_UNITS
from models.bayesian_lstm import BayesianLSTM

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE          = os.path.join(os.path.dirname(__file__), '..')
ARTIFACT_PATH  = os.path.join(_BASE, 'artifacts', 'artifacts.pkl')
SCALER_PATH    = os.path.join(_BASE, 'artifacts', 'scaler.pkl')
MODEL_PATH     = os.path.join(_BASE, 'artifacts', 'model.pt')

# ── Design tokens ─────────────────────────────────────────────────────────────
BG          = "#0F1420"
SURFACE     = "#1A1F2E"
BORDER      = "rgba(255,255,255,0.08)"
TEXT_PRI    = "#E8EAF0"
TEXT_SEC    = "#8892A4"
ACCENT      = "#1E88E5"
C_SUCCESS   = "#00C48C"
C_WARN      = "#FFB020"
C_DANGER    = "#EF4444"

VITAL_COLOR = {
    "HR":      "#E57373",
    "Temp":    "#FF8A65",
    "WBC":     "#9575CD",
    "Lactate": "#FFB74D",
    "SBP":     "#4FC3F7",
    "RR":      "#81C784",
    "SpO2":    "#64B5F6",
}

VITAL_LABEL = {
    "HR": "Heart Rate", "Temp": "Temperature", "WBC": "WBC",
    "Lactate": "Lactate", "SBP": "Systolic BP", "RR": "Resp. Rate", "SpO2": "SpO₂",
}

VITAL_SLIDER = {
    "HR":      dict(min_value=40,   max_value=160,  value=78,   step=1,   format="%d"),
    "Temp":    dict(min_value=34.0, max_value=42.0, value=37.0, step=0.1, format="%.1f"),
    "WBC":     dict(min_value=1.0,  max_value=30.0, value=7.5,  step=0.5, format="%.1f"),
    "Lactate": dict(min_value=0.2,  max_value=15.0, value=1.0,  step=0.1, format="%.1f"),
    "SBP":     dict(min_value=55,   max_value=210,  value=122,  step=1,   format="%d"),
    "RR":      dict(min_value=6,    max_value=45,   value=16,   step=1,   format="%d"),
    "SpO2":    dict(min_value=70,   max_value=100,  value=98,   step=1,   format="%d"),
}

NORMAL_BASELINE = [78, 37.0, 7.5, 1.0, 122, 16, 98]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SepsisWatch",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

  section[data-testid="stSidebar"] > div {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] div {{ color: {TEXT_PRI} !important; }}

  .sw-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 8px;
  }}

  .risk-score {{
    font-size: 3.2rem;
    font-weight: 600;
    letter-spacing: -1px;
    line-height: 1;
  }}
  .risk-label {{
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 6px;
  }}
  .ci-text {{
    font-size: 0.75rem;
    color: {TEXT_SEC};
    margin-top: 8px;
    font-family: 'JetBrains Mono', monospace;
  }}

  .vital-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid {BORDER};
    font-size: 0.82rem;
  }}
  .vital-row:last-child {{ border-bottom: none; }}
  .vital-name  {{ color: {TEXT_SEC}; width: 80px; }}
  .vital-value {{ font-weight: 500; color: {TEXT_PRI}; text-align: right; }}
  .vital-unit  {{ color: {TEXT_SEC}; font-size: 0.72rem; width: 55px; text-align: right; }}
  .dot-normal  {{ width:8px; height:8px; border-radius:50%; background:{C_SUCCESS}; display:inline-block; margin-right:6px; }}
  .dot-warn    {{ width:8px; height:8px; border-radius:50%; background:{C_WARN};    display:inline-block; margin-right:6px; }}
  .dot-danger  {{ width:8px; height:8px; border-radius:50%; background:{C_DANGER};  display:inline-block; margin-right:6px; }}

  .alert-banner {{
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.85rem;
    font-weight: 500;
    color: {C_DANGER};
    margin-bottom: 12px;
  }}
  .info-banner {{
    background: rgba(30,136,229,0.08);
    border: 1px solid rgba(30,136,229,0.25);
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.82rem;
    color: {TEXT_SEC};
    margin-bottom: 12px;
  }}

  div[data-testid="metric-container"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 14px 16px;
  }}
</style>
""", unsafe_allow_html=True)


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if not os.path.exists(ARTIFACT_PATH):
        return None
    return joblib.load(ARTIFACT_PATH)


@st.cache_resource
def load_model_and_scaler():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        return None, None
    scaler = joblib.load(SCALER_PATH)
    model  = BayesianLSTM(input_size=len(VITALS), hidden_size=64, num_layers=2, dropout=0.3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model, scaler


# ── Helpers ───────────────────────────────────────────────────────────────────
def vital_status(vital, value):
    lo, hi = NORMAL_RANGES[vital]
    if vital == 'SpO2':
        if value < 90:  return 'danger'
        if value < 95:  return 'warn'
        return 'normal'
    dev = (value - lo) / (hi - lo + 1e-9)
    if dev < -0.3 or dev > 1.3: return 'danger'
    if dev < 0    or dev > 1:   return 'warn'
    return 'normal'


def risk_color(score):
    if score >= 0.55: return C_DANGER
    if score >= 0.30: return C_WARN
    return C_SUCCESS


def risk_label(score):
    if score >= 0.55: return "HIGH RISK"
    if score >= 0.30: return "MODERATE"
    return "LOW RISK"


def status_dot(status):
    return f'<span class="dot-{status}"></span>'


def render_risk_card(score, std, qsofa=None):
    color = risk_color(score)
    label = risk_label(score)
    ci_lo = max(0, score - 1.96 * std)
    ci_hi = min(1, score + 1.96 * std)
    qsofa_str = f'<div class="ci-text" style="margin-top:10px;">qSOFA: {qsofa}/3</div>' if qsofa is not None else ''
    st.markdown(f"""
    <div class="sw-card" style="text-align:center; border-color:{color}33;">
      <div style="font-size:0.7rem; color:{TEXT_SEC}; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:10px;">
        Sepsis Risk Score
      </div>
      <div class="risk-score" style="color:{color};">{score:.0%}</div>
      <div class="risk-label" style="color:{color};">{label}</div>
      <div class="ci-text">95% CI &nbsp; {ci_lo:.0%} – {ci_hi:.0%}</div>
      {qsofa_str}
    </div>
    """, unsafe_allow_html=True)


def render_vitals_table(vitals_row):
    rows_html = ""
    for v in VITALS:
        val    = vitals_row[v]
        status = vital_status(v, val)
        lo, hi = NORMAL_RANGES[v]
        rows_html += f"""
        <div class="vital-row">
          <span class="vital-name">{status_dot(status)}{VITAL_LABEL[v]}</span>
          <span class="vital-value">{val:.1f}</span>
          <span class="vital-unit">{VITAL_UNITS[v]}</span>
        </div>"""
    st.markdown(f'<div class="sw-card">{rows_html}</div>', unsafe_allow_html=True)


# ── Plot builders ─────────────────────────────────────────────────────────────
_PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=SURFACE,
    font=dict(family="Inter, sans-serif", color=TEXT_SEC, size=11),
)


def _axis_style():
    return dict(
        showgrid=True, gridcolor="rgba(255,255,255,0.05)",
        zeroline=False, showline=False,
        tickfont=dict(size=10, color=TEXT_SEC),
    )


def build_vitals_fig(patient_vitals, step):
    """6-panel subplot (3 rows × 2 cols) of vital signs up to `step`."""
    pairs = [("HR","Temp"), ("WBC","Lactate"), ("SBP","RR")]
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Heart Rate (bpm)", "Temperature (°C)",
            "WBC (k/µL)",       "Lactate (mmol/L)",
            "Systolic BP (mmHg)","Resp. Rate (/min)",
        ],
        vertical_spacing=0.16,
        horizontal_spacing=0.10,
    )

    t = np.arange(step)
    for row_i, (v1, v2) in enumerate(pairs):
        for col_i, v in enumerate([v1, v2]):
            r, c  = row_i + 1, col_i + 1
            vals  = patient_vitals[:step, VITALS.index(v)]
            lo, hi = NORMAL_RANGES[v]
            color  = VITAL_COLOR[v]

            # Normal range band
            fig.add_trace(go.Scatter(
                x=list(t) + list(t[::-1]),
                y=[hi] * step + [lo] * step,
                fill="toself",
                fillcolor="rgba(255,255,255,0.03)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ), row=r, col=c)

            # Vital trace
            fig.add_trace(go.Scatter(
                x=t, y=vals,
                mode="lines",
                line=dict(color=color, width=1.8),
                showlegend=False,
                hovertemplate=f"{v}: %{{y:.1f}} {VITAL_UNITS[v]}<extra></extra>",
            ), row=r, col=c)

            # Current-step marker
            if step > 0:
                fig.add_trace(go.Scatter(
                    x=[step - 1], y=[vals[-1]],
                    mode="markers",
                    marker=dict(color=color, size=7, symbol="circle",
                                line=dict(color="white", width=1)),
                    showlegend=False, hoverinfo="skip",
                ), row=r, col=c)

    fig.update_layout(
        height=480,
        **_PLOT_LAYOUT,
        margin=dict(l=8, r=8, t=55, b=8),
    )
    fig.update_xaxes(**_axis_style(), title_text="Hour", title_font_size=9)
    fig.update_yaxes(**_axis_style())
    # Shrink subplot title font
    for ann in fig.layout.annotations:
        ann.font.size = 11
        ann.font.color = TEXT_SEC
    return fig


def build_risk_fig(risk_mean, risk_std, qsofa_arr, step, threshold):
    t    = np.arange(step)
    mean = risk_mean[:step]
    std  = risk_std[:step]
    q    = qsofa_arr[:step] / 3.0

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=list(t) + list(t[::-1]),
        y=list(np.clip(mean + 1.96 * std, 0, 1)) + list(np.clip(mean - 1.96 * std, 0, 1)[::-1]),
        fill="toself",
        fillcolor="rgba(239,68,68,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI",
    ))

    # Risk trace
    fig.add_trace(go.Scatter(
        x=t, y=mean, mode="lines",
        line=dict(color=C_DANGER, width=2.2),
        name="Model risk score",
    ))

    # qSOFA normalised
    fig.add_trace(go.Scatter(
        x=t, y=q, mode="lines",
        line=dict(color=ACCENT, width=1.6, dash="dot"),
        name="qSOFA / 3",
    ))

    # Threshold
    fig.add_hline(
        y=threshold, line_dash="dash",
        line_color=C_WARN, line_width=1.2,
        annotation_text=f"Alert ({threshold:.2f})",
        annotation_position="bottom right",
        annotation_font=dict(color=C_WARN, size=10),
    )

    fig.update_layout(
        title=dict(text="Sepsis Risk Score Over Time", font=dict(size=13, color=TEXT_PRI)),
        height=240,
        **_PLOT_LAYOUT,
        margin=dict(l=8, r=8, t=44, b=8),
        yaxis=dict(range=[0, 1], title="Probability", **_axis_style()),
        xaxis=dict(title="Hour", **_axis_style()),
        legend=dict(orientation="h", y=1.15, x=0, font=dict(size=10, color=TEXT_SEC),
                    bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    )
    return fig


def build_auroc_fig(results):
    fpr, tpr, auc = results["fpr"], results["tpr"], results["auroc"]
    q_auc   = results["qsofa_auroc"]
    q_sens  = results["qsofa_metrics"]["sensitivity"]
    q_spec  = results["qsofa_metrics"]["specificity"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             line=dict(color="#444", dash="dash"),
                             name="Random (AUC=0.50)", showlegend=True))
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             line=dict(color=C_DANGER, width=2.2),
                             name=f"BayesianLSTM (AUC={auc:.3f})"))
    fig.add_trace(go.Scatter(x=[1 - q_spec], y=[q_sens], mode="markers",
                             marker=dict(color=ACCENT, size=12, symbol="diamond",
                                         line=dict(color="white", width=1)),
                             name=f"qSOFA ≥ 2 (AUC={q_auc:.3f})"))
    fig.update_layout(
        title=dict(text="ROC Curve — All Patients", font=dict(size=13, color=TEXT_PRI)),
        height=360,
        **_PLOT_LAYOUT,
        margin=dict(l=8, r=8, t=50, b=8),
        xaxis=dict(title="False Positive Rate", range=[0,1], **_axis_style()),
        yaxis=dict(title="True Positive Rate",  range=[0,1], **_axis_style()),
        legend=dict(font=dict(size=10, color=TEXT_SEC), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def build_lead_fig(lead_times):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=lead_times, nbinsx=18,
        marker_color=C_DANGER, opacity=0.75,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=TEXT_PRI, line_width=1.2,
                  annotation_text="Clinical onset",
                  annotation_font=dict(color=TEXT_SEC, size=10))
    fig.update_layout(
        title=dict(text="Lead Time Before Clinical Onset", font=dict(size=13, color=TEXT_PRI)),
        height=260,
        **_PLOT_LAYOUT,
        margin=dict(l=8, r=8, t=50, b=8),
        xaxis=dict(title="Hours (positive = before onset)", **_axis_style()),
        yaxis=dict(title="Patients", **_axis_style()),
        showlegend=False,
    )
    return fig


def build_custom_risk_gauge(score, std):
    """Bullet-style gauge for the custom patient page."""
    ci_lo = max(0, score - 1.96 * std)
    ci_hi = min(1, score + 1.96 * std)
    color = risk_color(score)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number=dict(suffix="%", font=dict(size=42, color=color, family="Inter")),
        gauge=dict(
            axis=dict(range=[0, 100], ticksuffix="%",
                      tickfont=dict(size=10, color=TEXT_SEC)),
            bar=dict(color=color, thickness=0.25),
            bgcolor=SURFACE,
            borderwidth=0,
            steps=[
                dict(range=[0, 30],  color="rgba(0,196,140,0.10)"),
                dict(range=[30, 55], color="rgba(255,176,32,0.10)"),
                dict(range=[55, 100],color="rgba(239,68,68,0.10)"),
            ],
            threshold=dict(line=dict(color=C_WARN, width=2), thickness=0.7, value=50),
        ),
    ))
    fig.update_layout(
        height=240,
        **_PLOT_LAYOUT,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


# ── Pages ─────────────────────────────────────────────────────────────────────
def page_simulation(artifacts, threshold):
    data_df   = artifacts["data_df"]
    meta_df   = artifacts["meta_df"]
    risk_mean = artifacts["risk_mean"]
    risk_std  = artifacts["risk_std"]
    qsofa_2d  = artifacts["qsofa_2d"]
    n_patients = len(meta_df)
    n_hours    = risk_mean.shape[1]

    # Sidebar patient controls
    with st.sidebar:
        st.markdown("---")
        st.markdown("##### Patient")
        pid = st.number_input("Patient ID", 0, n_patients - 1,
                              value=st.session_state.get("pid", 0), step=1,
                              key="pid_input")
        st.session_state["pid"] = int(pid)

        c1, c2 = st.columns(2)
        if c1.button("Septic"):
            ids = meta_df[meta_df["is_septic"]]["patient_id"].tolist()
            st.session_state["pid"] = int(np.random.choice(ids))
            st.rerun()
        if c2.button("Healthy"):
            ids = meta_df[~meta_df["is_septic"]]["patient_id"].tolist()
            st.session_state["pid"] = int(np.random.choice(ids))
            st.rerun()

    pid     = st.session_state["pid"]
    prow    = meta_df[meta_df["patient_id"] == pid].iloc[0]
    pdata   = data_df[data_df["patient_id"] == pid].sort_values("time")
    p_vitals = pdata[VITALS].values
    p_risk   = risk_mean[pid]
    p_std    = risk_std[pid]
    p_qsofa  = qsofa_2d[pid]

    # Time scrubber
    step = st.slider(
        "Observation hour", min_value=1, max_value=n_hours,
        value=n_hours, step=1,
        help="Drag to reveal patient data up to this hour.",
    )

    current_risk = float(p_risk[step - 1])
    current_std  = float(p_std[step - 1])

    # Alert banner
    if current_risk >= threshold:
        st.markdown(
            f'<div class="alert-banner">⚠ SEPSIS ALERT — Risk {current_risk:.0%} exceeds threshold {threshold:.0%}</div>',
            unsafe_allow_html=True,
        )

    # Main layout: left = risk card + vitals table | right = charts
    left, right = st.columns([1, 2.6], gap="medium")

    with left:
        render_risk_card(current_risk, current_std, qsofa=int(p_qsofa[step - 1]))
        vitals_row = {v: float(p_vitals[step - 1, VITALS.index(v)]) for v in VITALS}
        render_vitals_table(vitals_row)

        if step == n_hours:
            truth_color = C_DANGER if prow["is_septic"] else C_SUCCESS
            truth_label = "SEPTIC" if prow["is_septic"] else "NOT SEPTIC"
            onset_str   = f" · onset h{int(prow['onset_hour'])}" if prow["is_septic"] else ""
            st.markdown(
                f'<div style="text-align:center; color:{truth_color}; font-size:0.8rem; '
                f'font-weight:600; letter-spacing:0.1em; border:1px solid {truth_color}33; '
                f'border-radius:8px; padding:8px; margin-top:4px;">'
                f'GROUND TRUTH: {truth_label}{onset_str}</div>',
                unsafe_allow_html=True,
            )

    with right:
        st.plotly_chart(build_vitals_fig(p_vitals, step), use_container_width=True)
        st.plotly_chart(build_risk_fig(p_risk, p_std, p_qsofa, step, threshold),
                        use_container_width=True)


def page_custom_patient(model, scaler):
    if model is None or scaler is None:
        st.error("Model artifacts not found. Run `python train.py` first.")
        return

    st.markdown("#### Enter Patient Vitals")
    st.markdown(
        '<div class="info-banner">Adjust the sliders to match the patient\'s current readings. '
        'The model assumes stable baseline physiology leading up to this observation.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.4, 1], gap="large")

    user_vitals = {}
    with left:
        st.markdown('<div class="sw-card">', unsafe_allow_html=True)
        for v in VITALS:
            kw = VITAL_SLIDER[v]
            lo, hi = NORMAL_RANGES[v]
            val = st.slider(
                f"{VITAL_LABEL[v]} ({VITAL_UNITS[v]})",
                key=f"cust_{v}", **kw,
            )
            user_vitals[v] = val
        st.markdown('</div>', unsafe_allow_html=True)

    # Compute risk in real time
    raw_seq = np.array([NORMAL_BASELINE] * 47 + [[user_vitals[v] for v in VITALS]],
                       dtype=np.float32)
    scaled_seq = scaler.transform(raw_seq)
    x_tensor   = torch.tensor(scaled_seq[np.newaxis], dtype=torch.float32)
    mean_arr, std_arr = model.mc_predict(x_tensor, n_samples=80)
    score = float(mean_arr[0, -1])
    std   = float(std_arr[0, -1])

    with right:
        render_risk_card(score, std)
        st.plotly_chart(build_custom_risk_gauge(score, std), use_container_width=True)

        # Abnormal vital summary
        abnormal = []
        for v in VITALS:
            status = vital_status(v, user_vitals[v])
            if status != "normal":
                lo, hi = NORMAL_RANGES[v]
                direction = "↑" if user_vitals[v] > hi else "↓"
                label_map = {"warn": C_WARN, "danger": C_DANGER}
                abnormal.append((v, direction, status, label_map[status]))

        if abnormal:
            st.markdown('<div class="sw-card"><div style="font-size:0.72rem; color:{TEXT_SEC}; '
                        'text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">'
                        'Abnormal Vitals</div>'.format(TEXT_SEC=TEXT_SEC), unsafe_allow_html=True)
            for v, direction, status, color in abnormal:
                st.markdown(
                    f'<div style="font-size:0.82rem; color:{color}; padding:2px 0;">'
                    f'{status_dot(status)}{VITAL_LABEL[v]} {direction} — {user_vitals[v]:.1f} {VITAL_UNITS[v]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="sw-card" style="text-align:center; color:{C_SUCCESS}; font-size:0.82rem;">'
                f'{status_dot("normal")} All vitals within normal range</div>',
                unsafe_allow_html=True,
            )


def page_performance(results):
    mm = results["model_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model AUROC",  f"{results['auroc']:.3f}")
    c2.metric("qSOFA AUROC",  f"{results['qsofa_auroc']:.3f}",
              delta=f"+{results['auroc'] - results['qsofa_auroc']:.3f}",
              delta_color="normal")
    c3.metric("Sensitivity",  f"{mm['sensitivity']:.1%}")
    c4.metric("Specificity",  f"{mm['specificity']:.1%}")

    lead = results["lead_times"]
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg Precision", f"{results['avg_precision']:.3f}")
    c6.metric("PPV",           f"{mm['PPV']:.1%}")
    c7.metric("NPV",           f"{mm['NPV']:.1%}")
    if lead:
        c8.metric("Median Lead Time", f"{np.median(lead):.1f} h")

    st.markdown("<br>", unsafe_allow_html=True)
    lc, rc = st.columns(2)
    with lc:
        st.plotly_chart(build_auroc_fig(results), use_container_width=True)
    with rc:
        if lead:
            st.plotly_chart(build_lead_fig(lead), use_container_width=True)

    import pandas as pd
    st.markdown("#### Threshold-level Comparison")
    tbl = pd.DataFrame({
        "Model":        ["BayesianLSTM (this work)", "qSOFA ≥ 2 (clinical standard)"],
        "AUROC":        [f"{results['auroc']:.3f}", f"{results['qsofa_auroc']:.3f}"],
        "Sensitivity":  [f"{mm['sensitivity']:.3f}", f"{results['qsofa_metrics']['sensitivity']:.3f}"],
        "Specificity":  [f"{mm['specificity']:.3f}", f"{results['qsofa_metrics']['specificity']:.3f}"],
        "PPV":          [f"{mm['PPV']:.3f}", "—"],
        "NPV":          [f"{mm['NPV']:.3f}", "—"],
    }).set_index("Model")
    st.dataframe(tbl, use_container_width=True)

    with st.expander("Methodology", expanded=False):
        st.markdown("""
        **Dataset** — 600 synthetic ICU patients (35% septic), 48-hour trajectories.
        Vital signs simulated via Ornstein-Uhlenbeck processes parameterised from published
        clinical reference ranges. Septic patients exhibit a sigmoid-shaped physiological drift
        toward sepsis targets beginning at a randomly sampled onset hour (10–32 h).

        **Model** — 2-layer LSTM (hidden dim 64) with MC Dropout (*p* = 0.3).
        Dropout is active during inference; 80 stochastic forward passes yield a
        posterior mean and 95% credible interval over the risk score at every time step.

        **Features** — Heart Rate, Temperature, WBC, Lactate, Systolic BP, Respiratory Rate, SpO₂.

        **Benchmark** — qSOFA (RR ≥ 22, SBP ≤ 100, altered mentation proxy via lactate > 2).

        **Training** — Adam, ReduceLROnPlateau, early stopping (patience 12).
        """)


# ── App shell ─────────────────────────────────────────────────────────────────
def main():
    artifacts        = load_artifacts()
    model, scaler    = load_model_and_scaler()

    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:12px 0 4px;">
          <div style="font-size:1.1rem; font-weight:600; color:{TEXT_PRI};">SepsisWatch</div>
          <div style="font-size:0.72rem; color:{TEXT_SEC};">Stanford CS153 · Early Sepsis Detection</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        page = st.radio(
            "Navigation",
            ["Patient Simulation", "Custom Patient", "Model Performance"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        threshold = st.slider("Alert threshold", 0.20, 0.80, 0.50, 0.05,
                              help="Risk score at which a sepsis alert fires.")

    st.markdown(f"""
    <div style="margin-bottom:20px;">
      <span style="font-size:1.4rem; font-weight:600; color:{TEXT_PRI};">SepsisWatch</span>
      <span style="margin-left:12px; font-size:0.78rem; color:{TEXT_SEC};">
        Early Sepsis Detection via Sequential Bayesian Inference on ICU Time Series
      </span>
    </div>
    """, unsafe_allow_html=True)

    if artifacts is None:
        st.error("No artifacts found. Run `python train.py` from the project root first.")
        st.code("cd ~/Downloads/sepsis_detection && python3 train.py")
        return

    if page == "Patient Simulation":
        page_simulation(artifacts, threshold)
    elif page == "Custom Patient":
        page_custom_patient(model, scaler)
    else:
        page_performance(artifacts["results"])


if __name__ == "__main__":
    main()
