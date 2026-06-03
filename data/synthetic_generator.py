"""
data/synthetic_generator.py
============================
Synthetic ICU patient data generator for HealthWatch.

Because access to real patient records (e.g. MIMIC-III) requires institutional
credentialing, this module generates physiologically realistic ICU vital-sign
trajectories using Ornstein-Uhlenbeck (OU) stochastic processes.  The OU
process is a mean-reverting random walk commonly used to model biological time
series: each new value is pulled toward a long-run mean (the patient's
individual baseline) while a stochastic noise term introduces realistic
moment-to-moment variation.

For septic patients a sigmoid-shaped drift is applied after a randomly sampled
onset hour, smoothly shifting the attractor from the patient's normal baseline
toward published sepsis physiological targets.

Reference ranges are drawn from:
  - Marino, P.L. (2014). The ICU Book, 4th ed. Lippincott Williams & Wilkins.
  - Seymour et al. (2016). JAMA 315(8):762-774 (qSOFA validation).
  - Dellinger et al. (2013). Surviving Sepsis Campaign guidelines.
"""

import numpy as np
import pandas as pd


# ── Vital sign identifiers (order must be consistent throughout the codebase) ──
VITALS = ["HR", "Temp", "WBC", "Lactate", "SBP", "RR", "SpO2"]

# ── Ornstein-Uhlenbeck parameters for normal (non-septic) physiology ──────────
# theta : mean-reversion speed  (higher = faster return to baseline)
# sigma : noise intensity        (higher = more moment-to-moment variation)
# min/max: hard physiological clamps applied after each step
NORMAL_PARAMS = {
    "HR":      {"mean": 78,   "std": 8,    "theta": 0.30, "sigma": 3.0,  "min": 40,  "max": 160},
    "Temp":    {"mean": 37.0, "std": 0.25, "theta": 0.20, "sigma": 0.10, "min": 34,  "max": 42 },
    "WBC":     {"mean": 7.5,  "std": 1.2,  "theta": 0.10, "sigma": 0.40, "min": 1,   "max": 30 },
    "Lactate": {"mean": 1.0,  "std": 0.2,  "theta": 0.20, "sigma": 0.08, "min": 0.2, "max": 15 },
    "SBP":     {"mean": 122,  "std": 10,   "theta": 0.30, "sigma": 4.0,  "min": 55,  "max": 210},
    "RR":      {"mean": 16,   "std": 2,    "theta": 0.30, "sigma": 0.8,  "min": 6,   "max": 45 },
    "SpO2":    {"mean": 98,   "std": 0.8,  "theta": 0.50, "sigma": 0.4,  "min": 70,  "max": 100},
}

# ── Physiological targets reached ~8 h after sepsis onset ────────────────────
# Values based on typical sepsis presentations described in Seymour et al. 2016
# and the Surviving Sepsis Campaign clinical criteria.
SEPSIS_TARGETS = {
    "HR":      {"mean": 118,  "std": 14},   # tachycardia
    "Temp":    {"mean": 38.9, "std": 0.5},  # fever
    "WBC":     {"mean": 15.0, "std": 3.0},  # leukocytosis
    "Lactate": {"mean": 3.8,  "std": 1.2},  # tissue hypoperfusion
    "SBP":     {"mean": 88,   "std": 10},   # hypotension
    "RR":      {"mean": 26,   "std": 3},    # tachypnea (qSOFA criterion)
    "SpO2":    {"mean": 92,   "std": 2},    # hypoxia
}

# ── Normal ranges used for colour-coded vital display in the dashboard ────────
NORMAL_RANGES = {
    "HR":      (60,   100),
    "Temp":    (36.5, 37.8),
    "WBC":     (4.0,  11.0),
    "Lactate": (0.5,  2.0),
    "SBP":     (100,  140),
    "RR":      (12,   20),
    "SpO2":    (95,   100),
}

# ── Display units for each vital ──────────────────────────────────────────────
VITAL_UNITS = {
    "HR": "bpm", "Temp": "°C", "WBC": "k/µL",
    "Lactate": "mmol/L", "SBP": "mmHg", "RR": "/min", "SpO2": "%",
}


def _ou_step(x: float, mu: float, theta: float, sigma: float, dt: float = 1.0) -> float:
    """
    Perform one discrete step of an Ornstein-Uhlenbeck process.

    The OU SDE is:  dX = θ(μ - X)dt + σ dW
    Discretised as: X_{t+1} = X_t + θ(μ - X_t)Δt + σ√Δt · ε,  ε ~ N(0,1)

    Args:
        x     : current value
        mu    : long-run mean (attractor)
        theta : mean-reversion speed
        sigma : noise intensity (diffusion coefficient)
        dt    : time step in hours (default 1)

    Returns:
        Next value X_{t+1}
    """
    return x + theta * (mu - x) * dt + sigma * np.sqrt(dt) * np.random.randn()


def generate_patient(
    n_hours: int = 48,
    is_septic: bool = False,
    onset_hour: int = None,
    rng: np.random.Generator = None,
) -> tuple:
    """
    Simulate one patient's vital-sign trajectory over n_hours.

    Each patient has a randomly sampled individual baseline (drawn from a
    narrower distribution around the population mean to model inter-patient
    variability).  For septic patients, vitals drift sigmoidally from the
    patient's baseline toward sepsis target values starting at onset_hour.

    Args:
        n_hours    : length of simulation in hours
        is_septic  : whether the patient develops sepsis
        onset_hour : hour at which sepsis begins (sampled if None)
        rng        : NumPy random Generator for reproducibility

    Returns:
        vitals     : np.ndarray of shape (n_hours, len(VITALS))
        labels     : np.ndarray of shape (n_hours,), 1.0 after onset
        onset_hour : int or None — actual onset hour used
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample onset hour uniformly between hours 10 and 32 of the 48-h window
    if is_septic and onset_hour is None:
        onset_hour = int(rng.integers(10, 32))

    vitals = np.zeros((n_hours, len(VITALS)), dtype=np.float64)
    labels = np.zeros(n_hours, dtype=np.float32)

    # Each patient's resting state is drawn with ±60 % of population std
    baseline = {
        v: float(rng.normal(NORMAL_PARAMS[v]["mean"], NORMAL_PARAMS[v]["std"] * 0.6))
        for v in VITALS
    }
    # Initialise time step 0 at baseline (clipped to physiological limits)
    for i, v in enumerate(VITALS):
        p = NORMAL_PARAMS[v]
        vitals[0, i] = np.clip(baseline[v], p["min"], p["max"])

    for t in range(1, n_hours):
        for i, v in enumerate(VITALS):
            p = NORMAL_PARAMS[v]

            if is_septic and t >= onset_hour:
                # Sigmoid weight α ∈ [0,1]: controls gradual drift to sepsis values.
                # Reaches ~0.5 at 4 h post-onset and ~0.99 at ~12 h post-onset.
                hours_since = t - onset_hour
                alpha = 1.0 / (1.0 + np.exp(-0.5 * (hours_since - 4)))
                target_mu = (1 - alpha) * baseline[v] + alpha * SEPSIS_TARGETS[v]["mean"]
                labels[t] = 1.0
            else:
                target_mu = baseline[v]

            raw = _ou_step(vitals[t - 1, i], target_mu, p["theta"], p["sigma"])
            vitals[t, i] = np.clip(raw, p["min"], p["max"])

    return vitals, labels, onset_hour


def compute_qsofa(df: pd.DataFrame) -> np.ndarray:
    """
    Compute qSOFA score (0–3) at each time step for a patient DataFrame.

    qSOFA criteria (Seymour et al., JAMA 2016):
      +1  Respiratory rate >= 22 /min
      +1  Systolic BP <= 100 mmHg
      +1  Altered mentation (approximated here by Lactate > 2.0 mmol/L
          as a proxy for tissue dysoxia; a limitation of vitals-only data)

    Args:
        df : DataFrame with columns matching VITALS and rows = time steps

    Returns:
        np.ndarray of integer scores, shape (n_hours,)
    """
    score = (
        (df["RR"]      >= 22  ).astype(int)
        + (df["SBP"]   <= 100 ).astype(int)
        + (df["Lactate"] > 2.0).astype(int)
    )
    return score.values


def generate_dataset(
    n_patients: int = 500,
    n_hours: int = 48,
    sepsis_rate: float = 0.35,
    seed: int = 42,
) -> tuple:
    """
    Generate a complete dataset of synthetic ICU patients.

    Args:
        n_patients  : total number of patients to simulate
        n_hours     : observation window per patient (hours)
        sepsis_rate : fraction of patients who develop sepsis
        seed        : random seed for reproducibility

    Returns:
        data_df : pd.DataFrame — long format, one row per (patient, hour),
                  columns: VITALS + [time, patient_id, label, is_septic, qSOFA]
        meta_df : pd.DataFrame — one row per patient,
                  columns: [patient_id, is_septic, onset_hour]
    """
    rng = np.random.default_rng(seed)

    # Shuffle septic/non-septic labels randomly
    n_septic = int(n_patients * sepsis_rate)
    is_septic_flags = np.array([True] * n_septic + [False] * (n_patients - n_septic))
    rng.shuffle(is_septic_flags)

    records  = []
    metadata = []

    for pid in range(n_patients):
        is_septic = bool(is_septic_flags[pid])
        vitals, labels, onset = generate_patient(
            n_hours=n_hours, is_septic=is_septic, rng=rng
        )

        # Build per-patient DataFrame and compute qSOFA at every time step
        df = pd.DataFrame(vitals, columns=VITALS)
        df["time"]       = np.arange(n_hours)
        df["patient_id"] = pid
        df["label"]      = labels
        df["is_septic"]  = is_septic
        df["qSOFA"]      = compute_qsofa(df)

        records.append(df)
        metadata.append({"patient_id": pid, "is_septic": is_septic, "onset_hour": onset})

    data_df = pd.concat(records, ignore_index=True)
    meta_df = pd.DataFrame(metadata)
    return data_df, meta_df
