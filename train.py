"""
train.py
=========
End-to-end training script for the HealthWatch Bayesian LSTM sepsis detector.

Run this script once before launching the dashboard.  It will:
  1. Generate a synthetic ICU dataset (600 patients × 48 hours).
  2. Split patients into train (80%) and validation (20%) sets.
  3. Fit a StandardScaler on training vital signs, then normalise both splits.
  4. Train a BayesianLSTM with BCE loss, Adam optimiser, and early stopping.
  5. Run Monte Carlo dropout inference on all 600 patients.
  6. Evaluate the model against qSOFA on the synthetic data.
  7. Save three artifacts to artifacts/:
       model.pt       — trained model weights (state_dict)
       scaler.pkl     — fitted StandardScaler (joblib)
       artifacts.pkl  — pre-computed MC predictions, qSOFA scores, and
                        evaluation results (joblib)

Usage
-----
    python train.py

Expected runtime: ~2–5 minutes on CPU.  CUDA is used automatically if available.

Architecture overview
---------------------
  Input:  (batch, 48, 7)  — 7 vital signs over 48 hours
  LSTM:   2 layers, hidden_size=64, inter-layer dropout=0.3
  Head:   Linear(64→32) → ReLU → Dropout(0.3) → Linear(32→1) → Sigmoid
  Output: (batch, 48)    — risk probability at each time step

Training details
----------------
  Loss function : Binary cross-entropy (BCELoss)
                  Plain BCE is used because the model's final layer already
                  applies sigmoid activation.
  Class balance : The dataset has ~35% sepsis prevalence.  No explicit
                  pos_weight is used in the final BCE loss; the synthetic
                  data separates cleanly so class weighting is not critical.
  Early stopping: Patience = 12 epochs on validation loss.
  LR schedule   : ReduceLROnPlateau (factor=0.5, patience=5).
  Gradient clip : Max-norm 1.0 per batch to stabilise training.

Positional indexing note
------------------------
Patient IDs from generate_dataset() are globally unique (0–599), but after
the train/val split the positional index within each subset diverges from the
patient_id column.  All tensor arrays (X, y) use a positional counter `i`
rather than patient_id as the row index to avoid IndexError.
"""

import os
import sys
import time
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Allow top-level imports regardless of the working directory
sys.path.insert(0, os.path.dirname(__file__))

from data.synthetic_generator import generate_dataset, VITALS
from models.bayesian_lstm import BayesianLSTM
from evaluation.metrics import evaluate_all

# ── Paths ──────────────────────────────────────────────────────────────────────
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

# ── Hyperparameters ────────────────────────────────────────────────────────────
# Data generation
N_PATIENTS  = 600       # total synthetic patients (35% septic = 210 cases)
N_HOURS     = 48        # hours of observation per patient
SEPSIS_RATE = 0.35      # fraction of patients who develop sepsis

# Model architecture
HIDDEN_SIZE = 64        # LSTM hidden dimension
NUM_LAYERS  = 2         # stacked LSTM layers
DROPOUT     = 0.3       # dropout rate (inter-layer + classification head)

# Training
BATCH_SIZE  = 32        # mini-batch size
EPOCHS      = 60        # maximum training epochs
LR          = 1e-3      # initial Adam learning rate
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ────────────────────────────────────────────────────────────────────

class SepsisDataset(Dataset):
    """
    PyTorch Dataset wrapping pre-processed vital-sign sequences and labels.

    Converts NumPy arrays to float32 tensors at construction time so that
    DataLoader workers do not perform repeated conversion during training.

    Args:
        X : np.ndarray of shape (n_patients, N_HOURS, n_features)
        y : np.ndarray of shape (n_patients, N_HOURS) — binary labels
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx]


# ── Data preparation ───────────────────────────────────────────────────────────

def prepare_tensors(
    data_df,
    meta_df,
    scaler=None,
    fit_scaler: bool = False,
) -> tuple:
    """
    Convert a long-format patient DataFrame into (X, y) tensor arrays.

    Stacks each patient's hourly observations into a 3-D array and optionally
    fits and applies a StandardScaler across all (time × patient) rows.

    Positional indexing: rows in X and y correspond to the i-th row of meta_df
    in iteration order — NOT to the patient_id values, which may not be
    contiguous after a train/val split.

    Args:
        data_df     : Long-format DataFrame (one row per patient-hour) with
                      columns VITALS + [time, patient_id, label].
        meta_df     : Patient-level DataFrame with [patient_id, is_septic, ...].
        scaler      : Pre-fitted StandardScaler.  Required when fit_scaler=False.
        fit_scaler  : If True, fit a new StandardScaler on this split's data.

    Returns:
        X_scaled : np.ndarray (n_patients, N_HOURS, n_features), float32
        y        : np.ndarray (n_patients, N_HOURS), float32
        scaler   : StandardScaler — the fitted scaler (new or passed-through)
    """
    n_patients = len(meta_df)
    X = np.zeros((n_patients, N_HOURS, len(VITALS)), dtype=np.float32)
    y = np.zeros((n_patients, N_HOURS), dtype=np.float32)

    # Populate arrays using positional counter i to avoid patient_id index mismatch
    for i, (_, row) in enumerate(meta_df.iterrows()):
        pid   = int(row["patient_id"])
        pdata = data_df[data_df["patient_id"] == pid].sort_values("time")
        X[i]  = pdata[VITALS].values
        y[i]  = pdata["label"].values

    # Normalise: flatten to (n_patients * N_HOURS, n_features), scale, reshape back
    flat = X.reshape(-1, len(VITALS))
    if fit_scaler:
        scaler = StandardScaler().fit(flat)

    X_scaled = scaler.transform(flat).reshape(n_patients, N_HOURS, len(VITALS))
    return X_scaled.astype(np.float32), y, scaler


# ── Training loop helpers ──────────────────────────────────────────────────────

def train_epoch(
    model: BayesianLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    pos_weight: torch.Tensor,
) -> float:
    """
    Run one epoch of mini-batch gradient descent.

    Gradient norms are clipped to 1.0 per batch to prevent exploding
    gradients in deep LSTMs on longer sequences.

    Args:
        model      : BayesianLSTM in train() mode.
        loader     : DataLoader over the training SepsisDataset.
        optimizer  : Adam optimiser.
        criterion  : BCELoss instance.
        pos_weight : Class weight tensor (computed but not used in plain BCE —
                     retained in the signature for interface flexibility).

    Returns:
        mean_loss : float — dataset-weighted average BCE loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)                              # (batch, N_HOURS)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(xb)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
    model: BayesianLSTM,
    loader: DataLoader,
    criterion: nn.Module,
) -> float:
    """
    Evaluate the model on a held-out split without gradient computation.

    The model is set to eval() mode so dropout is disabled and the
    prediction is deterministic.  This is appropriate for validation loss
    comparison (used for early stopping and LR scheduling).

    Args:
        model     : BayesianLSTM.
        loader    : DataLoader over the validation SepsisDataset.
        criterion : BCELoss instance.

    Returns:
        mean_loss : float — dataset-weighted average BCE loss.
    """
    model.eval()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred   = model(xb)
        total_loss += criterion(pred, yb).item() * len(xb)
    return total_loss / len(loader.dataset)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    """
    Full training pipeline: data generation → model training → evaluation → save.
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # ── 1. Synthetic data ──────────────────────────────────────────────────────
    print("Generating synthetic ICU dataset …")
    t0 = time.time()
    data_df, meta_df = generate_dataset(
        n_patients=N_PATIENTS,
        n_hours=N_HOURS,
        sepsis_rate=SEPSIS_RATE,
        seed=42,
    )
    print(f"  {N_PATIENTS} patients, {N_HOURS}h each  ({time.time()-t0:.1f}s)")
    print(f"  Septic: {meta_df['is_septic'].sum()}  |  "
          f"Non-septic: {(~meta_df['is_septic']).sum()}")

    # ── 2. Patient-level train / validation split ──────────────────────────────
    # Stratify on sepsis status to preserve the 35% class ratio in both splits.
    train_meta, val_meta = train_test_split(
        meta_df,
        test_size=0.2,
        stratify=meta_df["is_septic"],
        random_state=42,
    )
    # Reset index so positional enumeration in prepare_tensors is contiguous
    train_meta = train_meta.reset_index(drop=True)
    val_meta   = val_meta.reset_index(drop=True)

    # ── 3. Feature engineering + normalisation ─────────────────────────────────
    X_train, y_train, scaler = prepare_tensors(data_df, train_meta, fit_scaler=True)
    X_val,   y_val,   _      = prepare_tensors(data_df, val_meta,   scaler=scaler)

    train_loader = DataLoader(
        SepsisDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        SepsisDataset(X_val, y_val), batch_size=BATCH_SIZE
    )

    # ── 4. Model, loss, optimiser ──────────────────────────────────────────────
    # pos_weight is computed for interface completeness; BCELoss is used without it
    # because the synthetic data separates cleanly at the chosen prevalence.
    pos_weight = torch.tensor([(1 - SEPSIS_RATE) / SEPSIS_RATE]).to(DEVICE)
    criterion  = nn.BCELoss()   # sigmoid already applied inside BayesianLSTM.head

    model = BayesianLSTM(
        input_size=len(VITALS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    # ── 5. Training loop with early stopping ───────────────────────────────────
    print(f"\nTraining BayesianLSTM on {DEVICE} …")
    best_val_loss = float("inf")
    best_state    = None
    patience_cnt  = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, pos_weight)
        vl_loss = eval_epoch(model,   val_loader,   criterion)
        scheduler.step(vl_loss)

        star = ""
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            # Deep-copy state so best weights are not overwritten later
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
            star = " *"
        else:
            patience_cnt += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  "
                  f"train={tr_loss:.4f}  val={vl_loss:.4f}{star}")

        if patience_cnt >= 12:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best checkpoint
    model.load_state_dict(best_state)

    # ── 6. Monte Carlo inference on all patients ───────────────────────────────
    # The dashboard loads these pre-computed arrays to avoid re-running MC
    # dropout at request time for historical trend visualisation.
    print("\nRunning MC inference for all patients …")
    X_all, y_all, _ = prepare_tensors(data_df, meta_df, scaler=scaler)
    X_tensor = torch.tensor(X_all).to(DEVICE)

    # Process in batches to avoid GPU / RAM out-of-memory
    mean_list, std_list = [], []
    bs = 64
    for i in range(0, len(X_tensor), bs):
        m, s = model.mc_predict(X_tensor[i : i + bs], n_samples=50)
        mean_list.append(m)
        std_list.append(s)

    risk_mean = np.concatenate(mean_list, axis=0)   # (N_PATIENTS, N_HOURS)
    risk_std  = np.concatenate(std_list,  axis=0)   # (N_PATIENTS, N_HOURS)

    # ── 7. Build qSOFA + label arrays in meta_df positional order ─────────────
    qsofa_2d  = np.zeros((N_PATIENTS, N_HOURS), dtype=int)
    labels_2d = np.zeros((N_PATIENTS, N_HOURS), dtype=float)

    for i, (_, row) in enumerate(meta_df.iterrows()):
        pid   = int(row["patient_id"])
        pdata = data_df[data_df["patient_id"] == pid].sort_values("time")
        qsofa_2d[i]  = pdata["qSOFA"].values
        labels_2d[i] = pdata["label"].values

    # ── 8. Evaluation ──────────────────────────────────────────────────────────
    print("\nEvaluating …")
    results = evaluate_all(meta_df, risk_mean, labels_2d, qsofa_2d, threshold=0.5)

    print(f"  Model AUROC : {results['auroc']:.3f}")
    print(f"  qSOFA AUROC : {results['qsofa_auroc']:.3f}")
    mm = results["model_metrics"]
    print(f"  Sensitivity : {mm['sensitivity']:.3f}")
    print(f"  Specificity : {mm['specificity']:.3f}")
    if results["lead_times"]:
        print(f"  Median lead time vs. clinical onset: "
              f"{np.median(results['lead_times']):.1f} h")

    # ── 9. Save artifacts ──────────────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "model.pt"))
    joblib.dump(scaler,            os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    joblib.dump(
        {
            "data_df":    data_df,    # full long-format dataset
            "meta_df":    meta_df,    # patient-level metadata
            "risk_mean":  risk_mean,  # LSTM posterior mean (N_PATIENTS × N_HOURS)
            "risk_std":   risk_std,   # LSTM posterior std
            "qsofa_2d":   qsofa_2d,   # qSOFA scores
            "labels_2d":  labels_2d,  # ground-truth sepsis labels
            "results":    results,    # evaluation metrics dict
        },
        os.path.join(ARTIFACT_DIR, "artifacts.pkl"),
    )

    print(f"\nArtifacts saved to {ARTIFACT_DIR}/")
    print("Launch the dashboard with:  streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
