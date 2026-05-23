import numpy as np
import pandas as pd

VITALS = ['HR', 'Temp', 'WBC', 'Lactate', 'SBP', 'RR', 'SpO2']

# Ornstein-Uhlenbeck parameters for normal physiology
NORMAL_PARAMS = {
    'HR':      {'mean': 78,   'std': 8,   'theta': 0.30, 'sigma': 3.0,  'min': 40,  'max': 160},
    'Temp':    {'mean': 37.0, 'std': 0.25,'theta': 0.20, 'sigma': 0.10, 'min': 34,  'max': 42},
    'WBC':     {'mean': 7.5,  'std': 1.2, 'theta': 0.10, 'sigma': 0.40, 'min': 1,   'max': 30},
    'Lactate': {'mean': 1.0,  'std': 0.2, 'theta': 0.20, 'sigma': 0.08, 'min': 0.2, 'max': 15},
    'SBP':     {'mean': 122,  'std': 10,  'theta': 0.30, 'sigma': 4.0,  'min': 55,  'max': 210},
    'RR':      {'mean': 16,   'std': 2,   'theta': 0.30, 'sigma': 0.8,  'min': 6,   'max': 45},
    'SpO2':    {'mean': 98,   'std': 0.8, 'theta': 0.50, 'sigma': 0.4,  'min': 70,  'max': 100},
}

# Target physiological values during sepsis (reached after ~8h of onset)
SEPSIS_TARGETS = {
    'HR':      {'mean': 118,  'std': 14},
    'Temp':    {'mean': 38.9, 'std': 0.5},
    'WBC':     {'mean': 15.0, 'std': 3.0},
    'Lactate': {'mean': 3.8,  'std': 1.2},
    'SBP':     {'mean': 88,   'std': 10},
    'RR':      {'mean': 26,   'std': 3},
    'SpO2':    {'mean': 92,   'std': 2},
}

# Normal ranges for clinical coloring in the dashboard
NORMAL_RANGES = {
    'HR':      (60, 100),
    'Temp':    (36.5, 37.8),
    'WBC':     (4.0, 11.0),
    'Lactate': (0.5, 2.0),
    'SBP':     (100, 140),
    'RR':      (12, 20),
    'SpO2':    (95, 100),
}

VITAL_UNITS = {
    'HR': 'bpm', 'Temp': '°C', 'WBC': 'k/µL',
    'Lactate': 'mmol/L', 'SBP': 'mmHg', 'RR': '/min', 'SpO2': '%',
}


def _ou_step(x, mu, theta, sigma, dt=1.0):
    return x + theta * (mu - x) * dt + sigma * np.sqrt(dt) * np.random.randn()


def generate_patient(n_hours=48, is_septic=False, onset_hour=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if is_septic and onset_hour is None:
        onset_hour = int(rng.integers(10, 32))

    vitals = np.zeros((n_hours, len(VITALS)))
    labels = np.zeros(n_hours, dtype=np.float32)

    # Patient-specific baseline (inter-patient variability)
    baseline = {}
    for i, v in enumerate(VITALS):
        p = NORMAL_PARAMS[v]
        baseline[v] = float(rng.normal(p['mean'], p['std'] * 0.6))
        vitals[0, i] = np.clip(baseline[v], p['min'], p['max'])

    for t in range(1, n_hours):
        for i, v in enumerate(VITALS):
            p = NORMAL_PARAMS[v]
            if is_septic and t >= onset_hour:
                hours_since = t - onset_hour
                # Sigmoid-shaped drift so onset is gradual, not instantaneous
                alpha = 1.0 / (1.0 + np.exp(-0.5 * (hours_since - 4)))
                target_mu = (1 - alpha) * baseline[v] + alpha * SEPSIS_TARGETS[v]['mean']
                labels[t] = 1.0
            else:
                target_mu = baseline[v]

            raw = _ou_step(vitals[t - 1, i], target_mu, p['theta'], p['sigma'])
            vitals[t, i] = np.clip(raw, p['min'], p['max'])

    return vitals, labels, onset_hour


def compute_qsofa(df):
    score = (
        (df['RR'] >= 22).astype(int)
        + (df['SBP'] <= 100).astype(int)
        + (df['Lactate'] > 2.0).astype(int)  # lactate as mentation proxy
    )
    return score.values


def generate_dataset(n_patients=500, n_hours=48, sepsis_rate=0.35, seed=42):
    rng = np.random.default_rng(seed)

    n_septic = int(n_patients * sepsis_rate)
    is_septic_flags = np.array([True] * n_septic + [False] * (n_patients - n_septic))
    rng.shuffle(is_septic_flags)

    records = []
    metadata = []

    for pid in range(n_patients):
        is_septic = bool(is_septic_flags[pid])
        vitals, labels, onset = generate_patient(
            n_hours=n_hours, is_septic=is_septic, rng=rng
        )
        df = pd.DataFrame(vitals, columns=VITALS)
        df['time'] = np.arange(n_hours)
        df['patient_id'] = pid
        df['label'] = labels
        df['is_septic'] = is_septic
        df['qSOFA'] = compute_qsofa(df)
        records.append(df)
        metadata.append({'patient_id': pid, 'is_septic': is_septic, 'onset_hour': onset})

    data_df = pd.concat(records, ignore_index=True)
    meta_df = pd.DataFrame(metadata)
    return data_df, meta_df
