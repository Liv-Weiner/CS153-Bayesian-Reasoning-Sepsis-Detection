"""
models/symptom_engine.py
=========================
Symptom-based clinical assessment engine for HealthWatch.

Overview
--------
This module implements a weighted symptom-to-condition scoring system covering
14 conditions across a 39-symptom catalogue grouped into six anatomical /
physiological categories.  It is intentionally rule-based rather than
statistical, so every weight can be traced to its clinical rationale.

Scoring philosophy
------------------
Each condition has a ``weights`` dict mapping symptom keys to base point values.
A clinician reading the weights should be able to recognise them as reflecting
established pathognomonic relationships — e.g. neck stiffness carries the
highest single weight for meningitis (5 pts) because nuchal rigidity is a
cardinal sign; right lower quadrant pain carries the highest weight for
appendicitis for the same reason.

Severity multiplier
~~~~~~~~~~~~~~~~~~~
Symptom severity is reported on a 1–10 scale.  The raw weight is multiplied by:
  * 0.5  for mild symptoms     (severity 1–3)
  * 1.0  for moderate symptoms (severity 4–6)   [default]
  * 1.5  for severe symptoms   (severity 7–10)

Risk factors
~~~~~~~~~~~~
Medical history, family history, and travel/exposure context boost the raw
score via ``RISK_BONUSES`` before the confidence ratio is computed.
Example: a patient with known pregnancy receives +5.0 bonus toward preeclampsia
and +2.0 toward PE — both independently elevated by pregnancy.

Confidence normalisation
~~~~~~~~~~~~~~~~~~~~~~~~
``confidence = min(1.0, total / condition_threshold)``

Conditions are suppressed (not shown) below confidence < 0.12 to avoid noise.

Conditions covered (14)
-----------------------
Critical: Sepsis, Acute MI / ACS, Stroke / TIA, Meningitis / Encephalitis,
          Preeclampsia / Eclampsia, DKA, Pulmonary Embolism
High:     Pneumonia, Malaria, Tuberculosis, Typhoid Fever, Dengue Fever,
          Severe Anaemia, Appendicitis

Free-text parsing
-----------------
``parse_text()`` performs case-insensitive substring matching against
``TEXT_KEYWORDS``, a curated synonym dictionary of clinical and lay terms for
each symptom.  Matched symptoms are pre-filled as checkboxes in the UI.

Clinical basis
--------------
Condition selection reflects the leading causes of mortality in low- and
middle-income countries (LMICs) as reported by WHO Global Health Estimates
(2019) and the Global Burden of Disease study.  Symptom weights and red-flag
criteria are derived from the clinical guidelines cited in the dashboard
Citations tab (see condition_rules.py and app/dashboard.py for full references).
"""

# ── Symptom catalogue ─────────────────────────────────────────────────────────
# Symptoms are grouped into six anatomical / physiological domains so that
# the dashboard can render them in a logical, clinician-friendly layout.
# Keys are short snake_case identifiers used throughout this module.
# Values are human-readable labels shown in the UI.

SYMPTOM_GROUPS = {
    "Constitutional": {
        "fever":         "Fever (temp > 38 °C / 100.4 °F)",
        "chills":        "Chills or rigors",
        "sweating":      "Profuse daytime sweating",
        "night_sweats":  "Night sweats",
        "weakness":      "Fatigue or significant weakness",
        "weight_loss":   "Unintentional weight loss",
        "loss_appetite": "Loss of appetite",
    },
    "Cardiovascular": {
        "chest_pain":    "Chest pain or pressure",
        "palpitations":  "Racing or fast heartbeat",
        "low_bp":        "Known or reported low blood pressure",
        "pallor":        "Pale skin or mucous membranes",
        "edema":         "Swollen legs, ankles, or feet",
        "calf_pain":     "Calf pain or swelling",
    },
    "Respiratory": {
        "cough":         "Cough (dry or productive)",
        "dyspnea":       "Shortness of breath",
        "rapid_breath":  "Visibly rapid breathing",
        "hemoptysis":    "Coughing up blood",
    },
    "Gastrointestinal": {
        "nausea_vom":    "Nausea or vomiting",
        "abd_pain":      "Abdominal pain (general)",
        "abd_rlq":       "Pain worse at lower right abdomen",
        "diarrhea":      "Diarrhea (loose or watery stools)",
        "blood_stool":   "Blood in stool or black tarry stools",
        "jaundice":      "Skin or eye yellowing (jaundice)",
    },
    "Neurological": {
        "confusion":     "Confusion or altered mental status",
        "severe_ha":     "Sudden severe headache (worst of life)",
        "facial_droop":  "Facial drooping — one side",
        "limb_weak":     "One-sided arm or leg weakness",
        "speech_diff":   "Difficulty speaking or understanding",
        "dizziness":     "Dizziness or lightheadedness",
        "syncope":       "Fainting or loss of consciousness",
        "vision_change": "Vision changes or blurred vision",
    },
    "Urinary & Other": {
        "low_urine":     "Decreased urine output",
        "dysuria":       "Painful or burning urination",
        "back_pain":     "Back or flank pain",
        "rash":          "Skin rash (especially petechial / non-blanching)",
        "joint_pain":    "Joint or muscle pain",
        "neck_stiff":    "Neck stiffness",
        "photophobia":   "Sensitivity to light",
        "eye_pain":      "Eye pain or redness",
    },
}

# Flat lookup: symptom_key → display label (used in the severity-rating UI)
SYMPTOM_LABELS = {k: v for g in SYMPTOM_GROUPS.values() for k, v in g.items()}


# ── Condition definitions ─────────────────────────────────────────────────────
# Each condition entry contains:
#   name        : human-readable display name
#   urgency     : CRITICAL / HIGH / MODERATE — shown as a badge in the UI
#   description : one-sentence clinical summary
#   weights     : {symptom_key: base_points} — reflects pathognomonic strength
#   threshold   : denominator for confidence = total_score / threshold
#                 Set so that a patient with the full classic presentation
#                 scores approximately 1.0 (full confidence).
#   tests       : ordered list of recommended investigations
#   red_flags   : clinical findings requiring immediate escalation

CONDITIONS = {
    "sepsis": {
        "name": "Sepsis",
        "urgency": "CRITICAL",
        "description": "Life-threatening organ dysfunction from dysregulated response to infection.",
        "weights": {
            # SOFA / qSOFA criteria (Singer et al., JAMA 2016; Seymour et al., JAMA 2016)
            "fever": 3, "chills": 2, "rapid_breath": 3, "confusion": 3,
            "low_bp": 3, "low_urine": 3, "palpitations": 2, "sweating": 1, "weakness": 2,
        },
        # Classic presentation (fever + hypotension + altered mentation + oliguria) = 12 pts
        "threshold": 8,
        "tests": [
            "Blood cultures ×2 (before antibiotics)",
            "CBC with differential",
            "Serum lactate",
            "Procalcitonin",
            "Basic metabolic panel (Na, K, Creatinine, BUN)",
            "Urinalysis + urine culture",
            "Chest X-ray",
        ],
        "red_flags": [
            "Altered mental status or new confusion",
            "Systolic BP < 90 mmHg despite IV fluids",
            "Lactate >= 4 mmol/L",
            "Urine output < 0.5 mL/kg/h",
        ],
    },
    "pneumonia": {
        "name": "Pneumonia",
        "urgency": "HIGH",
        "description": "Lower respiratory tract infection; can progress rapidly to respiratory failure.",
        "weights": {
            # CURB-65 components adapted for symptom presentation (Lim et al., Thorax 2003)
            "cough": 3, "fever": 2, "dyspnea": 3, "rapid_breath": 3,
            "chest_pain": 2, "weakness": 1, "nausea_vom": 1, "hemoptysis": 2,
        },
        "threshold": 6,
        "tests": [
            "Chest X-ray (PA + lateral)",
            "Pulse oximetry",
            "CBC",
            "Sputum Gram stain + culture",
            "Blood cultures",
            "Urine Legionella + Pneumococcal antigen",
        ],
        "red_flags": [
            "SpO2 < 92% on room air",
            "Respiratory rate > 30/min",
            "Systolic BP < 90 mmHg",
        ],
    },
    "acute_mi": {
        "name": "Acute MI / ACS",
        "urgency": "CRITICAL",
        "description": "Acute coronary syndrome. ECG and serial troponin required for diagnosis.",
        "weights": {
            # Chest pain is the primary discriminator (HEART score, Backus et al. 2010)
            "chest_pain": 5, "palpitations": 2, "dyspnea": 3, "sweating": 2,
            "nausea_vom": 2, "weakness": 1, "low_bp": 2, "pallor": 1, "syncope": 2,
        },
        "threshold": 5,
        "tests": [
            "12-lead ECG — STAT (within 10 min)",
            "Troponin I or T — serial at 0h and 3-6h",
            "CK-MB",
            "Basic metabolic panel",
            "CBC",
            "Chest X-ray",
        ],
        "red_flags": [
            "Diaphoresis with chest pain",
            "Systolic BP < 90 mmHg (cardiogenic shock)",
            "New-onset arrhythmia or syncope",
            "Symptom onset < 12 h (reperfusion window)",
        ],
    },
    "stroke": {
        "name": "Stroke / TIA",
        "urgency": "CRITICAL",
        "description": "Cerebrovascular accident — time is brain, act within minutes.",
        "weights": {
            # FAST criteria: Face droop, Arm weakness, Speech difficulty (Nor et al., Lancet Neurology 2004)
            "facial_droop": 5, "limb_weak": 5, "speech_diff": 5,
            "severe_ha": 3, "confusion": 3, "nausea_vom": 1,
            "vision_change": 3, "dizziness": 2,
        },
        "threshold": 5,
        "tests": [
            "Non-contrast CT head — STAT",
            "Blood glucose (immediately)",
            "CBC + coagulation (PT / INR / aPTT)",
            "Basic metabolic panel",
            "12-lead ECG",
            "Blood pressure (both arms)",
        ],
        "red_flags": [
            "FAST: Face droop + Arm weakness + Speech difficulty + Time",
            "Symptom onset < 4.5 h (IV tPA eligibility window)",
            "BP > 220/120 mmHg",
            "Blood glucose < 50 or > 400 mg/dL",
        ],
    },
    "malaria": {
        "name": "Malaria",
        "urgency": "HIGH",
        "description": "Plasmodium infection — endemic in sub-Saharan Africa, SE Asia, South America.",
        "weights": {
            # Cyclical fever + rigors + sweating is the classic malaria triad
            "fever": 3, "chills": 3, "sweating": 3, "joint_pain": 2,
            "severe_ha": 2, "nausea_vom": 2, "weakness": 2, "confusion": 3,
        },
        # Full triad + 2 supporting symptoms ≈ threshold
        "threshold": 8,
        "tests": [
            "Malaria rapid diagnostic test (RDT)",
            "Thick and thin blood smear",
            "CBC (assess anaemia + thrombocytopenia)",
            "Blood glucose",
            "Basic metabolic panel",
        ],
        "red_flags": [
            "Cerebral malaria — seizures or coma",
            "Severe anaemia (Hgb < 7 g/dL)",
            "Respiratory distress",
            "Hypoglycaemia",
        ],
    },
    "meningitis": {
        "name": "Meningitis / Encephalitis",
        "urgency": "CRITICAL",
        "description": "Infection of meninges or brain — can be fatal within hours if untreated.",
        "weights": {
            # Kernig / Brudzinski signs captured by neck_stiff + photophobia
            "severe_ha": 4, "fever": 3, "neck_stiff": 5,
            "photophobia": 4, "confusion": 4, "nausea_vom": 2, "rash": 3,
        },
        "threshold": 7,
        "tests": [
            "CT head before LP (if focal neuro signs or papilloedema)",
            "Lumbar puncture — opening pressure, cell count, glucose, protein, culture",
            "Blood cultures x2",
            "CBC + differential",
            "CRP + procalcitonin",
        ],
        "red_flags": [
            "Non-blanching petechial rash (meningococcal)",
            "Rapidly declining GCS",
            "Seizures",
            "Signs of raised ICP (Cushing triad)",
        ],
    },
    "tuberculosis": {
        "name": "Tuberculosis (TB)",
        "urgency": "HIGH",
        "description": "Chronic mycobacterial infection — leading infectious disease killer globally.",
        "weights": {
            # Classic triad: chronic cough + night sweats + weight loss (Lawn & Zumla, Lancet 2011)
            "cough": 3, "night_sweats": 4, "weight_loss": 4, "fever": 2,
            "hemoptysis": 4, "weakness": 2, "loss_appetite": 2,
        },
        # Higher threshold: TB requires the full constitutional picture to flag
        "threshold": 9,
        "tests": [
            "Sputum AFB smear x3",
            "GeneXpert MTB/RIF (preferred if available)",
            "Chest X-ray (upper lobe cavities, infiltrates)",
            "TST (Mantoux) or IGRA (QuantiFERON)",
            "CBC + ESR + CRP",
        ],
        "red_flags": [
            "Haemoptysis (coughing blood)",
            "Miliary TB pattern on CXR",
            "TB meningitis — altered consciousness + neck stiffness",
            "HIV co-infection (greatly increases severity)",
        ],
    },
    "typhoid": {
        "name": "Typhoid Fever",
        "urgency": "HIGH",
        "description": "Salmonella typhi infection — common where sanitation is limited.",
        "weights": {
            # Stepwise fever + abdominal pain is the hallmark presentation (Bhutta, BMJ 2006)
            "fever": 3, "abd_pain": 3, "weakness": 2, "nausea_vom": 2,
            "severe_ha": 2, "diarrhea": 2, "loss_appetite": 2, "rash": 2,
        },
        "threshold": 9,
        "tests": [
            "Blood culture (gold standard — sensitivity ~60-80%)",
            "Widal test (limited specificity)",
            "CBC (relative leukopenia is characteristic)",
            "Stool + urine culture",
            "Liver function tests",
        ],
        "red_flags": [
            "Intestinal perforation — acute abdomen with peritonitis",
            "Intestinal haemorrhage",
            "Typhoid encephalopathy",
            "Relative bradycardia (pulse-temperature dissociation)",
        ],
    },
    "dengue": {
        "name": "Dengue Fever",
        "urgency": "HIGH",
        "description": "Arboviral infection spread by Aedes mosquitoes in tropical regions.",
        "weights": {
            # Retro-orbital pain (eye_pain) + arthralgia + rash are cardinal signs (WHO 2009)
            "fever": 3, "joint_pain": 4, "severe_ha": 3, "rash": 3,
            "nausea_vom": 2, "eye_pain": 3, "weakness": 2,
        },
        "threshold": 9,
        "tests": [
            "NS1 antigen test (days 1-5 of illness)",
            "Dengue IgM / IgG serology",
            "CBC (thrombocytopenia and leukopenia characteristic)",
            "Liver enzymes (ALT/AST)",
            "Haematocrit (rising Hct indicates plasma leakage)",
        ],
        "red_flags": [
            "Severe thrombocytopenia (platelets < 20,000)",
            "Bleeding — gums, nose, haematemesis",
            "Dengue haemorrhagic fever / shock syndrome",
            "Abdominal pain and vomiting (warning signs of progression)",
        ],
    },
    "anemia": {
        "name": "Severe Anaemia",
        "urgency": "MODERATE",
        "description": "Critical reduction in red blood cells or haemoglobin — common in LMIC.",
        "weights": {
            # Pallor is the most visible bedside sign; dyspnea + palpitations = compensatory
            "weakness": 3, "pallor": 4, "dyspnea": 3, "palpitations": 2,
            "dizziness": 3, "edema": 1, "loss_appetite": 1,
        },
        "threshold": 8,
        "tests": [
            "CBC with differential (Hgb, MCV, MCH)",
            "Peripheral blood smear",
            "Iron studies (ferritin, TIBC, serum iron)",
            "Reticulocyte count",
            "Vitamin B12 and folate levels",
            "Stool for occult blood",
        ],
        "red_flags": [
            "Hgb < 7 g/dL — transfusion threshold",
            "Cardiac decompensation (HR > 120, dyspnoea at rest)",
            "Acute blood loss source",
            "Haemolytic crisis (sickle cell, G6PD)",
        ],
    },
    "preeclampsia": {
        "name": "Preeclampsia / Eclampsia",
        "urgency": "CRITICAL",
        "description": "Hypertensive disorder of pregnancy — major cause of maternal mortality.",
        "weights": {
            # Visual disturbances + severe headache = pre-seizure warning signs (ACOG 2013)
            "severe_ha": 3, "vision_change": 4, "edema": 3,
            "abd_pain": 2, "nausea_vom": 1, "confusion": 3,
        },
        # Lower threshold because even moderate scores in a pregnant patient are serious
        "threshold": 6,
        "tests": [
            "Blood pressure measurement (diagnosis: BP >= 140/90 x2)",
            "Urine dipstick for protein (then 24h protein if positive)",
            "CBC (platelet count)",
            "Liver function tests (LFT)",
            "Creatinine",
            "Fetal monitoring (CTG, ultrasound for growth)",
        ],
        "red_flags": [
            "BP >= 160/110 mmHg (severe range)",
            "Seizures (eclampsia) — airway, breathing, circulation",
            "Platelet < 100,000 (HELLP syndrome)",
            "Severe epigastric / right upper quadrant pain (liver)",
        ],
    },
    "appendicitis": {
        "name": "Appendicitis",
        "urgency": "HIGH",
        "description": "Inflammation of the appendix — surgical emergency if perforated.",
        "weights": {
            # RLQ pain migration is the dominant discriminator (Alvarado score, 1986)
            "abd_rlq": 5, "abd_pain": 3, "nausea_vom": 2,
            "fever": 2, "loss_appetite": 2,
        },
        "threshold": 7,
        "tests": [
            "Abdominal ultrasound (non-invasive first-line)",
            "CBC (leukocytosis in ~80%)",
            "CRP",
            "CT abdomen/pelvis with contrast (if diagnosis uncertain)",
            "Urinalysis (exclude UTI / renal colic)",
            "Pregnancy test (females of reproductive age)",
        ],
        "red_flags": [
            "Peritonism — rigid abdomen, rebound tenderness",
            "Fever > 39 C with abdominal rigidity (perforation)",
            "WBC > 18,000",
        ],
    },
    "dka": {
        "name": "Diabetic Ketoacidosis (DKA)",
        "urgency": "CRITICAL",
        "description": "Life-threatening diabetic emergency from insulin deficiency.",
        "weights": {
            # Kussmaul respiration (rapid_breath) + abdominal pain are hallmarks (ADA 2009)
            "nausea_vom": 2, "abd_pain": 2, "weakness": 2,
            "rapid_breath": 3, "confusion": 2, "loss_appetite": 1,
            "low_urine": 1,
        },
        "threshold": 7,
        "tests": [
            "Blood glucose (typically > 250 mg/dL)",
            "Urine or serum ketones",
            "Arterial blood gas (pH, bicarbonate)",
            "Basic metabolic panel — Na, K, Cr, bicarb",
            "CBC",
            "HbA1c",
        ],
        "red_flags": [
            "pH < 7.0 (severe acidosis)",
            "Altered consciousness",
            "K < 3.5 mEq/L (risk of fatal arrhythmia with insulin)",
            "Hyperosmolarity (> 320 mOsm/kg)",
        ],
    },
    "pe": {
        "name": "Pulmonary Embolism (PE)",
        "urgency": "CRITICAL",
        "description": "Blood clot in pulmonary artery — medical emergency.",
        "weights": {
            # Wells PE score components: DVT signs (calf_pain), haemoptysis, pleuritic chest pain
            "dyspnea": 4, "chest_pain": 3, "palpitations": 2, "syncope": 3,
            "hemoptysis": 2, "calf_pain": 3, "edema": 1,
        },
        "threshold": 7,
        "tests": [
            "D-dimer (high sensitivity if low pre-test probability)",
            "CT pulmonary angiography (CTPA) — diagnostic standard",
            "ECG (sinus tachycardia, S1Q3T3 pattern)",
            "Troponin + BNP (risk stratification)",
            "Lower limb Doppler ultrasound (detect DVT)",
            "CXR (usually normal, but exclude other causes)",
        ],
        "red_flags": [
            "Haemodynamic instability — SBP < 90 mmHg (massive PE)",
            "SpO2 < 90% despite supplemental oxygen",
            "Syncope or cardiac arrest",
            "Right heart strain on ECG or echo (risk of death)",
        ],
    },
}


# ── Risk factor catalogue ─────────────────────────────────────────────────────
# Three categories of risk factors collected in the UI:
#   Prior Medical Conditions — personal chronic diseases and procedural history
#   Family History           — first-degree relative disease history
#   Travel & Exposure        — recent geographic and environmental exposures
#
# Keys follow the convention: rf_* (medical), fh_* (family), tr_* (travel)

RISK_FACTOR_GROUPS = {
    "Prior Medical Conditions": {
        "rf_diabetes":   "Diabetes (Type 1 or 2)",
        "rf_htn":        "Hypertension (high blood pressure)",
        "rf_heart":      "Chronic heart disease or prior MI",
        "rf_hiv":        "HIV / AIDS",
        "rf_tb":         "Previous tuberculosis",
        "rf_cancer":     "Active cancer or recent chemotherapy",
        "rf_sickle":     "Sickle cell disease or trait",
        "rf_copd":       "COPD or chronic lung disease",
        "rf_ckd":        "Chronic kidney disease",
        "rf_pregnancy":  "Currently pregnant",
        "rf_surgery":    "Surgery or major procedure in last 4 weeks",
        "rf_immobility": "Prolonged immobility or long travel (> 4 h)",
    },
    "Family History": {
        "fh_heart":      "Heart attack or heart disease",
        "fh_stroke":     "Stroke",
        "fh_diabetes":   "Diabetes",
        "fh_cancer":     "Cancer",
        "fh_sickle":     "Sickle cell disease",
    },
    "Travel & Exposure": {
        "tr_malaria":    "Recent travel to malaria-endemic region",
        "tr_typhoid":    "Recent travel to typhoid/cholera-endemic region",
        "tr_tb":         "Known contact with active tuberculosis",
        "tr_water":      "Exposure to potentially contaminated water/food",
    },
}

# Flat lookup: risk_factor_key → display label
RISK_FACTOR_LABELS = {k: v for g in RISK_FACTOR_GROUPS.values() for k, v in g.items()}

# Bonus score added to specific conditions when a risk factor is present.
# Values are calibrated so that a single strong risk factor (e.g. pregnancy → PE +2,
# preeclampsia +5) can tip a moderate symptom score into the HIGH confidence tier.
# Sources: ACC/AHA (O'Gara 2013), ESC PE guidelines (Konstantinides 2020),
#          WHO TB guidelines (2022), ACOG preeclampsia guidelines (2013).
RISK_BONUSES = {
    "rf_diabetes":   {"acute_mi": 2.0, "stroke": 1.5, "sepsis": 1.0, "dka": 5.0},
    "rf_htn":        {"acute_mi": 2.0, "stroke": 2.0, "preeclampsia": 2.0},
    "rf_heart":      {"acute_mi": 3.0, "pe": 1.0},
    "rf_hiv":        {"tuberculosis": 3.0, "sepsis": 2.0, "meningitis": 2.0},
    "rf_tb":         {"tuberculosis": 2.5},
    "rf_cancer":     {"pe": 3.5, "sepsis": 2.0},
    "rf_sickle":     {"anemia": 4.0, "sepsis": 1.0, "stroke": 1.0},
    "rf_copd":       {"pneumonia": 2.0},
    "rf_ckd":        {"sepsis": 1.0},
    "rf_pregnancy":  {"preeclampsia": 5.0, "pe": 2.0, "anemia": 1.5},
    "rf_surgery":    {"pe": 3.0, "sepsis": 2.0},
    "rf_immobility": {"pe": 3.5},
    "fh_heart":      {"acute_mi": 1.5, "stroke": 1.0},
    "fh_stroke":     {"stroke": 1.5},
    "fh_diabetes":   {"dka": 1.0},
    "fh_sickle":     {"anemia": 2.0},
    "tr_malaria":    {"malaria": 4.0},
    "tr_typhoid":    {"typhoid": 4.0},
    "tr_tb":         {"tuberculosis": 3.0},
    "tr_water":      {"typhoid": 2.0},
}


# ── Free-text keyword dictionary ──────────────────────────────────────────────
# Maps each symptom key to a list of substrings (lowercase) that should trigger
# detection when they appear anywhere in the patient's free-text description.
# Each list includes both lay terms ("throwing up") and clinical terminology
# ("emesis") to handle diverse users — from patients describing their own symptoms
# to nurses entering bedside observations.
TEXT_KEYWORDS = {
    "fever":         ["fever", "temperature", "hot", "febrile", "pyrexia", "high temp"],
    "chills":        ["chill", "shiver", "rigor", "shaking"],
    "sweating":      ["sweat", "diaphoresis", "perspire"],
    "night_sweats":  ["night sweat", "sweating at night", "drenched at night"],
    "weakness":      ["weak", "fatigue", "tired", "exhausted", "lethargy", "no energy"],
    "weight_loss":   ["weight loss", "losing weight", "lost weight"],
    "loss_appetite": ["no appetite", "not eating", "loss of appetite", "anorexia"],
    "chest_pain":    ["chest pain", "chest pressure", "chest tight", "chest discomfort"],
    "palpitations":  ["palpitation", "racing heart", "fast heart", "heart pounding", "heart flutter"],
    "low_bp":        ["low blood pressure", "hypotension", "dizzy when standing", "low bp"],
    "pallor":        ["pale", "pallor", "pasty", "whitish"],
    "edema":         ["swollen", "edema", "oedema", "swelling", "puffy legs", "swollen ankles"],
    "calf_pain":     ["calf pain", "calf swelling", "leg clot", "dvt", "leg pain"],
    "cough":         ["cough", "coughing"],
    "dyspnea":       ["short of breath", "shortness of breath", "difficulty breathing",
                      "breathless", "can't breathe", "breathing hard"],
    "rapid_breath":  ["rapid breathing", "fast breathing", "tachypnea", "breathing fast"],
    "hemoptysis":    ["coughing blood", "blood in sputum", "hemoptysis", "spitting blood"],
    "nausea_vom":    ["nausea", "vomiting", "vomit", "sick to stomach", "throwing up", "nauseated"],
    "abd_pain":      ["abdominal pain", "stomach pain", "belly pain", "stomach ache", "tummy pain"],
    "abd_rlq":       ["lower right pain", "right lower quadrant", "rlq pain", "appendix"],
    "diarrhea":      ["diarrhea", "diarrhoea", "loose stool", "watery stool", "runny stool"],
    "blood_stool":   ["blood in stool", "bloody stool", "tarry stool", "melena", "rectal bleed"],
    "jaundice":      ["jaundice", "yellow skin", "yellow eyes", "icterus", "yellowing"],
    "confusion":     ["confused", "confusion", "disoriented", "not making sense", "altered"],
    "severe_ha":     ["headache", "head pain", "migraine", "head ache", "worst headache"],
    "facial_droop":  ["face droop", "facial droop", "drooping face", "mouth droop"],
    "limb_weak":     ["arm weak", "leg weak", "limb weak", "one-sided weak", "hemiplegia", "paralysis"],
    "speech_diff":   ["speech", "slurred", "can't speak", "aphasia", "difficulty speaking", "word finding"],
    "dizziness":     ["dizzy", "dizziness", "vertigo", "lightheaded", "light-headed"],
    "syncope":       ["faint", "fainting", "passed out", "loss of consciousness", "blackout", "syncope"],
    "vision_change": ["blurry vision", "blurred vision", "vision change", "double vision",
                      "visual disturbance", "can't see"],
    "low_urine":     ["decreased urine", "not urinating", "oliguria", "dark urine", "low urine output"],
    "dysuria":       ["painful urination", "burning urination", "dysuria", "pain when urinating"],
    "back_pain":     ["back pain", "flank pain", "kidney pain", "lower back"],
    "rash":          ["rash", "spots", "skin lesion", "petechiae", "hives", "skin eruption"],
    "joint_pain":    ["joint pain", "muscle pain", "myalgia", "arthralgia", "body ache", "aching"],
    "neck_stiff":    ["neck stiff", "stiff neck", "nuchal rigidity", "can't move neck"],
    "photophobia":   ["light sensitivity", "photophobia", "sensitive to light", "light hurts"],
    "eye_pain":      ["eye pain", "sore eye", "eye ache", "retro-orbital", "behind eye pain"],
}


# ── Public API ────────────────────────────────────────────────────────────────

def parse_text(text: str) -> list:
    """
    Extract symptom keys from a free-text clinical description.

    Performs case-insensitive substring matching against the ``TEXT_KEYWORDS``
    dictionary.  Each symptom is added at most once regardless of how many of
    its keywords appear.  The result is used to pre-populate the symptom
    checkboxes in the dashboard.

    Args:
        text : Free-text input from the clinician or patient.

    Returns:
        list of str — symptom keys detected in the text, in the order that
        entries appear in ``TEXT_KEYWORDS``.

    Example:
        >>> parse_text("Patient has high fever, chills, and neck stiffness")
        ['fever', 'chills', 'neck_stiff']
    """
    text_lower = text.lower()
    found = []
    for sym_key, keywords in TEXT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                if sym_key not in found:
                    found.append(sym_key)
                break  # stop checking keywords once this symptom is matched
    return found


def _severity_multiplier(s: int) -> float:
    """
    Convert a 1–10 severity rating to a score multiplier.

    The multiplier scales the base symptom weight to reflect how severely
    the patient is experiencing each symptom.

    Tiers:
        1–3   → 0.5  (mild — present but not limiting function)
        4–6   → 1.0  (moderate — default when severity is unknown)
        7–10  → 1.5  (severe — significantly impacts the patient)

    Args:
        s : Severity rating, integer in [1, 10].

    Returns:
        float multiplier: 0.5, 1.0, or 1.5.
    """
    if s <= 3:
        return 0.5
    if s <= 6:
        return 1.0
    return 1.5


def assess(
    selected: list,
    severities: dict = None,
    risk_factors: list = None,
) -> list:
    """
    Score all 14 conditions against the provided symptom and risk-factor profile.

    Algorithm
    ---------
    For each condition c:
      1. base_score  = Σ  weight(c, s) × severity_multiplier(severity(s))
                         for each selected symptom s
      2. rf_bonus    = Σ  RISK_BONUSES[rf][c]
                         for each selected risk factor rf
      3. total       = base_score + rf_bonus
      4. confidence  = min(1.0, total / condition_threshold)
      5. Suppress if confidence < 0.12  (too little signal — avoids noise)
      6. level       = HIGH (≥ 0.60) | MODERATE (≥ 0.35) | LOW

    Results are sorted by total score (descending).  The top result is
    expanded by default in the dashboard.

    Args:
        selected     : List of symptom keys that are present.
        severities   : dict {symptom_key: int 1-10}.  Defaults to 5 (moderate)
                       for any symptom not in the dict.
        risk_factors : List of risk-factor keys (rf_*, fh_*, tr_*).

    Returns:
        List of result dicts, each containing:
            id          : condition key (e.g. "sepsis")
            name        : display name
            urgency     : "CRITICAL" / "HIGH" / "MODERATE"
            description : one-sentence clinical summary
            confidence  : float in [0, 1]
            level       : "HIGH" / "MODERATE" / "LOW"
            tests       : list of recommended investigations
            red_flags   : list of escalation triggers
            score       : raw total score (base + rf_bonus)
            rf_bonus    : contribution from risk factors alone
        Returns an empty list if no symptoms or risk factors are provided.
    """
    if not selected and not risk_factors:
        return []

    if severities   is None:
        severities   = {}
    if risk_factors is None:
        risk_factors = []

    results = []
    for cid, cond in CONDITIONS.items():
        # Step 1 & 2: weighted symptom score + risk factor bonus
        base = sum(
            cond["weights"].get(s, 0) * _severity_multiplier(severities.get(s, 5))
            for s in selected
        )
        rf_bonus = sum(RISK_BONUSES.get(rf, {}).get(cid, 0) for rf in risk_factors)
        total    = base + rf_bonus

        if total == 0:
            continue

        # Step 4 & 5: normalise and filter noise
        confidence = min(1.0, total / cond["threshold"])
        if confidence < 0.12:
            continue

        level = "HIGH" if confidence >= 0.60 else "MODERATE" if confidence >= 0.35 else "LOW"

        results.append({
            "id":          cid,
            "name":        cond["name"],
            "urgency":     cond["urgency"],
            "description": cond["description"],
            "confidence":  confidence,
            "level":       level,
            "tests":       cond["tests"],
            "red_flags":   cond["red_flags"],
            "score":       total,
            "rf_bonus":    rf_bonus,
        })

    # Sort by raw score so the best-matching condition appears first
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
