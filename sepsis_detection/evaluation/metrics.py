import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix,
)


def compute_auroc(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return fpr, tpr, thresholds, auc


def compute_pr_curve(y_true, y_score):
    prec, rec, thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return prec, rec, thresholds, ap


def threshold_metrics(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    ppv = tp / (tp + fp + 1e-9)
    npv = tn / (tn + fn + 1e-9)
    return {
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
    }


def qsofa_metrics(y_true, qsofa_scores, threshold=2):
    """Treat qSOFA >= threshold as a positive alert."""
    y_pred = (qsofa_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    return {'sensitivity': sensitivity, 'specificity': specificity}


def time_to_detection(meta_df, risk_scores_2d, threshold=0.5, n_hours=48):
    """
    For each septic patient, compute how many hours before clinical onset
    the model first crosses the threshold.

    risk_scores_2d : (n_patients, n_hours) array
    Returns list of lead-time hours (positive = detected before onset).
    """
    lead_times = []
    for _, row in meta_df[meta_df['is_septic']].iterrows():
        pid = int(row['patient_id'])
        onset = row['onset_hour']
        if onset is None or np.isnan(onset):
            continue
        risk = risk_scores_2d[pid]
        alert_steps = np.where(risk >= threshold)[0]
        if len(alert_steps) == 0:
            continue
        first_alert = alert_steps[0]
        lead_times.append(int(onset) - int(first_alert))
    return lead_times


def evaluate_all(meta_df, risk_mean_2d, risk_labels_2d, qsofa_2d,
                 threshold=0.5, n_hours=48):
    """
    Aggregate evaluation across all patients using the last observed time step.
    """
    y_true, y_score, q_scores = [], [], []

    for i, (_, row) in enumerate(meta_df.iterrows()):
        y_true.append(int(row['is_septic']))
        y_score.append(float(risk_mean_2d[i, -1]))
        q_scores.append(int(qsofa_2d[i, -1]))

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    q_scores = np.array(q_scores)

    fpr, tpr, thresh, auc = compute_auroc(y_true, y_score)
    prec, rec, _, ap = compute_pr_curve(y_true, y_score)
    model_metrics = threshold_metrics(y_true, y_score, threshold)
    sofa_metrics = qsofa_metrics(y_true, q_scores, threshold=2)

    # qSOFA AUROC (treat score as continuous)
    qsofa_auc = roc_auc_score(y_true, q_scores)

    lead = time_to_detection(meta_df, risk_mean_2d, threshold=threshold)

    return {
        'fpr': fpr, 'tpr': tpr, 'thresholds': thresh, 'auroc': auc,
        'precision': prec, 'recall': rec, 'avg_precision': ap,
        'model_metrics': model_metrics,
        'qsofa_metrics': sofa_metrics,
        'qsofa_auroc': qsofa_auc,
        'lead_times': lead,
    }
