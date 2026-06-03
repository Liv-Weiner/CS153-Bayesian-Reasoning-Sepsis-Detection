"""
models/condition_rules.py
==========================
Rule-based clinical scoring for the Vital Signs Monitor.

Each function implements a simplified, vitals-only adaptation of an
established clinical scoring tool.  These are intentionally lightweight —
designed for pre-assessment in settings where laboratory results are not
yet available — and clearly labelled with their evidence source.

Scoring philosophy
------------------
Rather than exact replication of published scores (which often require
labs, imaging, or history), each function extracts the vital-sign
*components* of its source score and assigns points consistent with the
original weighting rationale.  The output probability is a normalised
score, NOT a calibrated posterior probability.

Sources
-------
- CURB-65  : Lim WS et al. Thorax 2003;58(5):377-382
- Shock Index : Allgower M, Burri C. DMW 1967;92(43):1947-1950
- ESC PE   : Konstantinides SV et al. Eur Heart J 2020;41(4):543-603
- ACC/AHA  : O'Gara PT et al. JACC 2013;61(4):e78-e140
"""


def score_pneumonia(vitals: dict) -> tuple:
    """
    Estimate pneumonia severity from vital signs.

    Adapted from CURB-65 (Lim et al., Thorax 2003).  Lab components
    (serum urea) and age are omitted; SpO2 is added as it is available
    at the bedside and strongly predicts outcome.

    Scoring:
      RR >= 30        : +2  (CURB-65 criterion, severe tachypnea)
      RR 22-29        : +1  (elevated but not severe)
      SBP < 90        : +2  (CURB-65 hypotension criterion)
      SpO2 < 90%      : +3  (severe hypoxia, not in original CURB-65)
      SpO2 90-91%     : +2
      SpO2 92-94%     : +1
      Temp >= 39.0°C  : +1  (high fever)
      Temp 38.3-38.9  : +0.5 (moderate fever / hypothermia)
      HR >= 120       : +0.5 (compensatory tachycardia)

    Risk thresholds (score / 6.0):
      < 2 pts  → LOW      (outpatient management may be appropriate)
      2-3 pts  → MODERATE (hospital admission recommended)
      >= 4 pts → HIGH     (consider ICU; CURB-65 >= 3 equivalent)

    Args:
        vitals : dict with keys HR, Temp, SBP, RR, SpO2

    Returns:
        probability : float in [0, 1], normalised score
        level       : str, one of "LOW" / "MODERATE" / "HIGH"
        findings    : list of str, human-readable abnormal findings
    """
    score    = 0.0
    findings = []

    rr   = vitals.get("RR",   16)
    sbp  = vitals.get("SBP",  120)
    spo2 = vitals.get("SpO2", 98)
    temp = vitals.get("Temp", 37.0)
    hr   = vitals.get("HR",   78)

    # Respiratory rate
    if rr >= 30:
        score += 2.0; findings.append(f"Severe tachypnea — RR {rr:.0f}/min (≥ 30)")
    elif rr >= 22:
        score += 1.0; findings.append(f"Tachypnea — RR {rr:.0f}/min")

    # Blood pressure
    if sbp < 90:
        score += 2.0; findings.append(f"Hypotension — SBP {sbp:.0f} mmHg")

    # Oxygenation (most clinically significant addition beyond CURB-65)
    if spo2 < 90:
        score += 3.0; findings.append(f"Severe hypoxia — SpO₂ {spo2:.0f}%")
    elif spo2 < 92:
        score += 2.0; findings.append(f"Significant hypoxia — SpO₂ {spo2:.0f}%")
    elif spo2 < 95:
        score += 1.0; findings.append(f"Mild hypoxia — SpO₂ {spo2:.0f}%")

    # Temperature
    if temp >= 39.0:
        score += 1.0; findings.append(f"High fever — {temp:.1f}°C")
    elif temp >= 38.3 or temp < 36.0:
        score += 0.5; findings.append(f"Fever / hypothermia — {temp:.1f}°C")

    # Heart rate
    if hr >= 120:
        score += 0.5; findings.append(f"Tachycardia — HR {hr:.0f} bpm")

    probability = min(1.0, score / 6.0)
    level = "HIGH" if score >= 4 else "MODERATE" if score >= 2 else "LOW"
    return probability, level, findings


def score_shock(vitals: dict) -> tuple:
    """
    Estimate hypovolemic / distributive shock severity using Shock Index.

    Primary metric: Shock Index (SI) = HR / SBP
      (Allgower & Burri, DMW 1967; validated by Rady et al., Resuscitation 1994)

      SI < 0.7   : normal
      SI 0.7-1.0 : borderline — Class I-II haemorrhage
      SI 1.0-1.4 : moderate   — Class III haemorrhage
      SI > 1.4   : severe     — Class IV haemorrhage / septic shock

    Supplementary points awarded for absolute hypotension and tachycardia,
    which independently predict poor outcome even when SI is borderline.

    Args:
        vitals : dict with keys HR, SBP, RR, SpO2

    Returns:
        probability : float in [0, 1]
        level       : str, "LOW" / "MODERATE" / "HIGH"
        findings    : list of str
    """
    score    = 0.0
    findings = []

    hr   = vitals.get("HR",   78)
    sbp  = vitals.get("SBP",  120)
    rr   = vitals.get("RR",   16)
    spo2 = vitals.get("SpO2", 98)

    # Shock Index — primary discriminator
    si = hr / max(sbp, 1)
    if si > 1.4:
        score += 4.0; findings.append(f"Severe shock index — {si:.2f} (> 1.4)")
    elif si > 1.0:
        score += 2.5; findings.append(f"Elevated shock index — {si:.2f} (> 1.0)")
    elif si > 0.7:
        score += 1.0; findings.append(f"Borderline shock index — {si:.2f}")

    # Absolute SBP thresholds (independent of HR)
    if sbp < 80:
        score += 2.0; findings.append(f"Critical hypotension — SBP {sbp:.0f} mmHg")
    elif sbp < 90:
        score += 1.5; findings.append(f"Hypotension — SBP {sbp:.0f} mmHg")
    elif sbp < 100:
        score += 0.5; findings.append(f"Low-normal BP — SBP {sbp:.0f} mmHg")

    # Heart rate severity tiers
    if hr > 140:
        score += 1.5; findings.append(f"Severe tachycardia — HR {hr:.0f} bpm")
    elif hr > 120:
        score += 1.0; findings.append(f"Tachycardia — HR {hr:.0f} bpm")
    elif hr > 100:
        score += 0.5; findings.append(f"Mild tachycardia — HR {hr:.0f} bpm")

    # Compensatory tachypnea and hypoxia
    if rr > 25:
        score += 0.5; findings.append(f"Tachypnea — RR {rr:.0f}/min")
    if spo2 < 92:
        score += 1.0; findings.append(f"Hypoxia — SpO₂ {spo2:.0f}%")

    probability = min(1.0, score / 7.0)
    level = "HIGH" if score >= 4 else "MODERATE" if score >= 2 else "LOW"
    return probability, level, findings


def score_cardiac(vitals: dict) -> tuple:
    """
    Identify haemodynamic patterns consistent with a cardiac emergency.

    Targets acute coronary syndrome (ACS), acute MI, and cardiogenic shock.
    Without ECG and troponin, only haemodynamic instability can be assessed.
    This function detects patterns of:
      - Cardiogenic shock: low SBP + compensatory tachycardia
      - Bradyarrhythmia:  very low HR (heart block, sick sinus)
      - Tachyarrhythmia:  very high HR (VT, AF with rapid ventricular response)
      - Pulmonary oedema: tachypnea + hypoxia with low-output state

    Clinical basis: ACC/AHA STEMI guidelines (O'Gara et al., JACC 2013);
    ESC ACS guidelines (Collet et al., Eur Heart J 2021).

    Args:
        vitals : dict with keys HR, SBP, RR, SpO2

    Returns:
        probability : float in [0, 1]
        level       : str, "LOW" / "MODERATE" / "HIGH"
        findings    : list of str
    """
    score    = 0.0
    findings = []

    hr   = vitals.get("HR",   78)
    sbp  = vitals.get("SBP",  120)
    rr   = vitals.get("RR",   16)
    spo2 = vitals.get("SpO2", 98)

    # Low BP — most important haemodynamic indicator of cardiogenic compromise
    if sbp < 90:
        score += 3.0; findings.append(f"Cardiogenic shock pattern — SBP {sbp:.0f} mmHg")
    elif sbp < 100:
        score += 1.5; findings.append(f"Low SBP — {sbp:.0f} mmHg")

    # Tachyarrhythmia (HR > 130) or bradyarrhythmia (HR < 50)
    if hr > 130:
        score += 1.5; findings.append(f"Tachyarrhythmia pattern — HR {hr:.0f} bpm")
    elif hr < 50:
        score += 2.0; findings.append(f"Bradycardia — possible heart block (HR {hr:.0f} bpm)")
    elif hr < 60:
        score += 0.5; findings.append(f"Low-normal HR — {hr:.0f} bpm")

    # Tachypnea suggests pulmonary oedema (raised left heart filling pressures)
    if rr >= 25:
        score += 1.0; findings.append(f"Tachypnea — possible pulmonary oedema (RR {rr:.0f}/min)")

    # Hypoxia in low-output state
    if spo2 < 92:
        score += 2.0; findings.append(f"Significant hypoxia — SpO₂ {spo2:.0f}%")
    elif spo2 < 95:
        score += 0.5; findings.append(f"Mild hypoxia — SpO₂ {spo2:.0f}%")

    probability = min(1.0, score / 6.0)
    level = "HIGH" if score >= 3 else "MODERATE" if score >= 1.5 else "LOW"
    return probability, level, findings


# ── Metadata displayed in the dashboard for each monitored condition ──────────
# 'model_note' is shown beneath the risk score to make the scoring method
# transparent to the clinician.
CONDITION_META = {
    "sepsis": {
        "label":      "Sepsis",
        "subtitle":   "Systemic infection causing organ dysfunction",
        "model_note": "Bayesian LSTM — trained on synthetic ICU data",
        "tests": [
            "Blood cultures ×2 (before antibiotics)",
            "CBC with differential",
            "Serum lactate",
            "Procalcitonin",
            "Basic metabolic panel (Na, K, Creatinine, BUN)",
            "Urinalysis + urine culture",
            "Chest X-ray",
        ],
        "actions": [
            "IV broad-spectrum antibiotics within 1 hour of suspicion",
            "IV fluid resuscitation — 30 mL/kg crystalloid",
            "Vasopressors if MAP < 65 mmHg persists after fluids",
            "Monitor urine output (target ≥ 0.5 mL/kg/h)",
        ],
    },
    "pneumonia": {
        "label":      "Severe Pneumonia",
        "subtitle":   "Lower respiratory infection with systemic involvement",
        "model_note": "Clinical rules — CURB-65 adapted (Lim et al., Thorax 2003)",
        "tests": [
            "Chest X-ray (PA + lateral)",
            "Pulse oximetry",
            "CBC",
            "Sputum Gram stain + culture",
            "Blood cultures",
            "Urine Legionella + Pneumococcal antigen",
        ],
        "actions": [
            "Supplemental O₂ (target SpO₂ ≥ 92%)",
            "Empiric antibiotics (community vs. hospital-acquired protocol)",
            "Consider ICU admission if CURB-65 ≥ 3",
        ],
    },
    "shock": {
        "label":      "Hypovolemic / Distributive Shock",
        "subtitle":   "Circulatory failure from volume loss or vasodilation",
        "model_note": "Clinical rules — Shock Index (Allgower & Burri, 1967)",
        "tests": [
            "CBC",
            "Basic metabolic panel",
            "Serum lactate",
            "Type & screen / crossmatch",
            "Coagulation panel (PT / INR / aPTT)",
            "FAST ultrasound (if available)",
        ],
        "actions": [
            "Large-bore IV access ×2",
            "IV fluid bolus 500–1,000 mL crystalloid",
            "Identify and control source (bleeding, infection, etc.)",
            "Consider vasopressors if unresponsive to fluids",
        ],
    },
    "cardiac": {
        "label":      "Cardiac Emergency (ACS / MI)",
        "subtitle":   "Acute coronary syndrome or cardiogenic shock",
        "model_note": "Clinical rules — haemodynamic pattern (ECG + troponin required for dx)",
        "tests": [
            "12-lead ECG — STAT (within 10 min of presentation)",
            "Troponin I or T — serial at 0h and 3–6h",
            "CK-MB",
            "Basic metabolic panel",
            "CBC",
            "Chest X-ray",
            "Echocardiogram (if available)",
        ],
        "actions": [
            "Aspirin 325 mg PO (if no contraindication)",
            "IV access + continuous cardiac monitoring",
            "Nitroglycerin for ongoing chest pain (only if SBP > 90)",
            "Urgent cardiology consult / transfer",
        ],
    },
}
