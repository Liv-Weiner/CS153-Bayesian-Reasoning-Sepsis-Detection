"""
evaluation/metrics.py
======================
Model evaluation utilities for the HealthWatch sepsis detection pipeline.

This module provides functions to assess the Bayesian LSTM's performance
against both standard binary classification metrics and the qSOFA clinical
baseline.  All functions operate on NumPy arrays and are designed to work
with the prediction arrays produced by ``train.py``.

Key metrics
-----------
- AUROC            : Area under the ROC curve (primary discriminative metric)
- Average Precision: Area under the precision-recall curve (better for
                     imbalanced classes — ~35% sepsis prevalence here)
- Sensitivity      : True positive rate at a fixed probability threshold
- Specificity      : True negative rate at a fixed probability threshold
- PPV / NPV        : Positive/Negative predictive values
- Lead time        : Hours before clinical onset that the model first exceeds
                     the alert threshold (most clinically meaningful metric)

qSOFA baseline
--------------
qSOFA (quick Sequential Organ Failure Assessment) is a validated 3-item
bedside score (Seymour et al., JAMA 2016).  Its AUROC and sensitivity /
specificity at threshold ≥ 2 are computed here as a comparison point for
the LSTM model.

Important note on evaluation scope
-----------------------------------
All metrics are computed on SYNTHETIC data.  The AUROC reported (≈ 1.000)
reflects the clean separation between OU trajectories for septic vs.
non-septic patients by design — it is NOT expected to replicate on real
ICU patient records.  See the dashboard Citations tab for a full transparency
statement.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix,
)


def compute_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple:
    """
    Compute the ROC curve and AUROC for binary classification.

    Args:
        y_true  : 1-D array of ground-truth binary labels (0 / 1).
        y_score : 1-D array of predicted probabilities in [0, 1].

    Returns:
        fpr        : np.ndarray — false positive rates at each threshold
        tpr        : np.ndarray — true positive rates at each threshold
        thresholds : np.ndarray — probability thresholds corresponding to
                     each (fpr, tpr) pair
        auc        : float — area under the ROC curve
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return fpr, tpr, thresholds, auc


def compute_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple:
    """
    Compute the precision-recall curve and average precision.

    Average precision (AP) summarises the PR curve as a weighted mean of
    precision at each recall step; it is more informative than AUROC when
    the positive class is rare (e.g. ~35% sepsis prevalence).

    Args:
        y_true  : 1-D array of ground-truth binary labels.
        y_score : 1-D array of predicted probabilities.

    Returns:
        precision  : np.ndarray — precision at each threshold
        recall     : np.ndarray — recall at each threshold
        thresholds : np.ndarray — probability thresholds
        ap         : float — average precision score
    """
    prec, rec, thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return prec, rec, thresholds, ap


def threshold_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute confusion-matrix-derived metrics at a fixed probability threshold.

    Args:
        y_true    : 1-D array of ground-truth binary labels.
        y_score   : 1-D array of predicted probabilities.
        threshold : Probability cutoff above which a prediction is treated
                    as a positive alert (default 0.5).

    Returns:
        dict with keys:
            threshold    : float — the cutoff used
            sensitivity  : float — TP / (TP + FN)   recall for the positive class
            specificity  : float — TN / (TN + FP)
            PPV          : float — TP / (TP + FP)   positive predictive value
            NPV          : float — TN / (TN + FN)   negative predictive value
            TP, FP, TN, FN : int — raw confusion-matrix counts

    Note:
        A small epsilon (1e-9) is added to all denominators to prevent
        division-by-zero when a class is absent from the predictions.
    """
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    ppv         = tp / (tp + fp + 1e-9)
    npv         = tn / (tn + fn + 1e-9)

    return {
        "threshold":   threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "PPV":         ppv,
        "NPV":         npv,
        "TP":  int(tp),
        "FP":  int(fp),
        "TN":  int(tn),
        "FN":  int(fn),
    }


def qsofa_metrics(
    y_true: np.ndarray,
    qsofa_scores: np.ndarray,
    threshold: int = 2,
) -> dict:
    """
    Evaluate qSOFA as a binary classifier at a given integer threshold.

    qSOFA score ≥ 2 is the recommended alert threshold from Seymour et al.
    (JAMA 2016).  This function treats any patient with qSOFA ≥ threshold as
    a positive sepsis screen and computes sensitivity and specificity against
    the ground-truth labels.

    Args:
        y_true       : 1-D array of ground-truth binary labels.
        qsofa_scores : 1-D integer array of qSOFA scores (0–3) at the last
                       observed time step for each patient.
        threshold    : Alert threshold (default 2 — per Seymour et al.).

    Returns:
        dict with keys:
            sensitivity : float — recall of qSOFA >= threshold
            specificity : float
    """
    y_pred = (qsofa_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    return {"sensitivity": sensitivity, "specificity": specificity}


def time_to_detection(
    meta_df: pd.DataFrame,
    risk_scores_2d: np.ndarray,
    threshold: float = 0.5,
    n_hours: int = 48,
) -> list:
    """
    Measure how many hours before clinical onset the model raises its first alert.

    For each septic patient the function finds the first time step at which the
    predicted risk crosses ``threshold``, then computes
    ``lead_time = onset_hour − first_alert_hour``.

    A positive lead time means the model alerted BEFORE the clinical onset
    (desirable — enables early intervention).  A negative lead time means the
    alert came AFTER onset.  Patients with no alert crossing are excluded.

    Args:
        meta_df        : Patient-level DataFrame with columns
                         [patient_id, is_septic, onset_hour].
        risk_scores_2d : np.ndarray of shape (n_patients, n_hours) — posterior
                         mean risk from ``model.mc_predict()``.
                         Row order must match the positional order of meta_df.
        threshold      : Risk threshold for declaring an alert (default 0.5).
        n_hours        : Sequence length (unused directly; retained for
                         interface clarity).

    Returns:
        list of int — lead times in hours for all septic patients who
        crossed the threshold.  Positive values indicate early detection.
    """
    lead_times = []
    for _, row in meta_df[meta_df["is_septic"]].iterrows():
        pid    = int(row["patient_id"])
        onset  = row["onset_hour"]
        if onset is None or np.isnan(onset):
            continue

        # risk_scores_2d is indexed positionally (row 0 = first patient in meta_df),
        # NOT by patient_id.  The row index for this patient must be looked up
        # from the positional index of meta_df at call time.
        risk         = risk_scores_2d[pid]
        alert_steps  = np.where(risk >= threshold)[0]
        if len(alert_steps) == 0:
            continue

        first_alert = alert_steps[0]
        lead_times.append(int(onset) - int(first_alert))

    return lead_times


def evaluate_all(
    meta_df: pd.DataFrame,
    risk_mean_2d: np.ndarray,
    risk_labels_2d: np.ndarray,
    qsofa_2d: np.ndarray,
    threshold: float = 0.5,
    n_hours: int = 48,
) -> dict:
    """
    Run the full evaluation suite across all patients.

    Uses the risk score at the final time step (index ``-1``) for each patient
    as the summary prediction.  This captures the full 48-hour trajectory that
    the LSTM has seen before making its end-of-window assessment.

    Row ordering: ``risk_mean_2d``, ``risk_labels_2d``, and ``qsofa_2d`` must
    all have the same positional order as ``meta_df`` (i.e. row i corresponds
    to the i-th row of meta_df, NOT necessarily patient_id == i).

    Args:
        meta_df        : Patient-level DataFrame with [patient_id, is_septic,
                         onset_hour].
        risk_mean_2d   : np.ndarray (n_patients, n_hours) — LSTM posterior mean.
        risk_labels_2d : np.ndarray (n_patients, n_hours) — ground-truth labels.
        qsofa_2d       : np.ndarray (n_patients, n_hours) — qSOFA integer scores.
        threshold      : Alert threshold for binary metrics (default 0.5).
        n_hours        : Observation window length (default 48).

    Returns:
        dict with keys:
            fpr, tpr, thresholds : ROC curve arrays
            auroc                : float — model AUROC
            precision, recall    : PR curve arrays
            avg_precision        : float — average precision (PR-AUC)
            model_metrics        : dict from threshold_metrics()
            qsofa_metrics        : dict from qsofa_metrics()
            qsofa_auroc          : float — qSOFA AUROC (treats score as continuous)
            lead_times           : list of int from time_to_detection()
    """
    y_true, y_score, q_scores = [], [], []

    # Extract the last time step (full 48-h sequence seen) for each patient.
    # Use positional enumerate to align with the positionally-ordered 2-D arrays.
    for i, (_, row) in enumerate(meta_df.iterrows()):
        y_true.append(int(row["is_septic"]))
        y_score.append(float(risk_mean_2d[i, -1]))
        q_scores.append(int(qsofa_2d[i, -1]))

    y_true   = np.array(y_true)
    y_score  = np.array(y_score)
    q_scores = np.array(q_scores)

    # Standard classification metrics
    fpr, tpr, thresh, auc = compute_auroc(y_true, y_score)
    prec, rec, _, ap       = compute_pr_curve(y_true, y_score)
    model_m                = threshold_metrics(y_true, y_score, threshold)
    sofa_m                 = qsofa_metrics(y_true, q_scores, threshold=2)

    # qSOFA AUROC (treat integer score 0-3 as a continuous predictor)
    qsofa_auc = roc_auc_score(y_true, q_scores)

    # Lead-time analysis for septic patients
    lead = time_to_detection(meta_df, risk_mean_2d, threshold=threshold)

    return {
        "fpr":           fpr,
        "tpr":           tpr,
        "thresholds":    thresh,
        "auroc":         auc,
        "precision":     prec,
        "recall":        rec,
        "avg_precision": ap,
        "model_metrics": model_m,
        "qsofa_metrics": sofa_m,
        "qsofa_auroc":   qsofa_auc,
        "lead_times":    lead,
    }
