"""
train.py — generates synthetic data, trains BayesianLSTM, saves artifacts.
Run once before launching the dashboard:
    python train.py
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

sys.path.insert(0, os.path.dirname(__file__))
from data.synthetic_generator import generate_dataset, VITALS
from models.bayesian_lstm import BayesianLSTM
from evaluation.metrics import evaluate_all

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
N_PATIENTS   = 600
N_HOURS      = 48
SEPSIS_RATE  = 0.35
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
DROPOUT      = 0.3
BATCH_SIZE   = 32
EPOCHS       = 60
LR           = 1e-3
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SepsisDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_tensors(data_df, meta_df, scaler=None, fit_scaler=False):
    n_patients = len(meta_df)
    X = np.zeros((n_patients, N_HOURS, len(VITALS)), dtype=np.float32)
    y = np.zeros((n_patients, N_HOURS), dtype=np.float32)

    for i, (_, row) in enumerate(meta_df.iterrows()):
        pid = int(row['patient_id'])
        pdata = data_df[data_df['patient_id'] == pid].sort_values('time')
        X[i] = pdata[VITALS].values
        y[i] = pdata['label'].values

    flat = X.reshape(-1, len(VITALS))
    if fit_scaler:
        scaler = StandardScaler().fit(flat)
    X_scaled = scaler.transform(flat).reshape(n_patients, N_HOURS, len(VITALS))
    return X_scaled.astype(np.float32), y, scaler


def train_epoch(model, loader, optimizer, criterion, pos_weight):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        total_loss += criterion(pred, yb).item() * len(xb)
    return total_loss / len(loader.dataset)


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    print("Generating synthetic ICU dataset …")
    t0 = time.time()
    data_df, meta_df = generate_dataset(
        n_patients=N_PATIENTS, n_hours=N_HOURS, sepsis_rate=SEPSIS_RATE, seed=42
    )
    print(f"  {N_PATIENTS} patients, {N_HOURS}h each  ({time.time()-t0:.1f}s)")
    print(f"  Septic: {meta_df['is_septic'].sum()}  |  Non-septic: {(~meta_df['is_septic']).sum()}")

    # Train / val split (patient-level)
    train_meta, val_meta = train_test_split(
        meta_df, test_size=0.2, stratify=meta_df['is_septic'], random_state=42
    )
    train_meta = train_meta.reset_index(drop=True)
    val_meta   = val_meta.reset_index(drop=True)

    X_train, y_train, scaler = prepare_tensors(data_df, train_meta, fit_scaler=True)
    X_val,   y_val,   _      = prepare_tensors(data_df, val_meta, scaler=scaler)

    train_loader = DataLoader(SepsisDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(SepsisDataset(X_val,   y_val),   batch_size=BATCH_SIZE)

    # Class-weighted BCE to handle imbalance
    pos_weight = torch.tensor([(1 - SEPSIS_RATE) / SEPSIS_RATE]).to(DEVICE)
    criterion  = nn.BCELoss()   # sigmoid already in model; use plain BCE

    model = BayesianLSTM(
        input_size=len(VITALS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print(f"\nTraining BayesianLSTM on {DEVICE} …")
    best_val_loss = float('inf')
    best_state    = None
    patience_cnt  = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, pos_weight)
        vl_loss = eval_epoch(model,   val_loader,   criterion)
        scheduler.step(vl_loss)

        star = ''
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
            star = ' *'
        else:
            patience_cnt += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  train={tr_loss:.4f}  val={vl_loss:.4f}{star}")

        if patience_cnt >= 12:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    # Pre-compute MC predictions for all patients (used by dashboard)
    print("\nRunning MC inference for all patients …")
    X_all, y_all, _ = prepare_tensors(data_df, meta_df, scaler=scaler)
    X_tensor = torch.tensor(X_all).to(DEVICE)

    # Process in batches to avoid OOM
    mean_list, std_list = [], []
    bs = 64
    for i in range(0, len(X_tensor), bs):
        m, s = model.mc_predict(X_tensor[i:i+bs], n_samples=50)
        mean_list.append(m)
        std_list.append(s)
    risk_mean = np.concatenate(mean_list, axis=0)  # (n_patients, n_hours)
    risk_std  = np.concatenate(std_list,  axis=0)

    # qSOFA per patient per time step — positional order matches X_all
    qsofa_2d = np.zeros((N_PATIENTS, N_HOURS), dtype=int)
    labels_2d = np.zeros((N_PATIENTS, N_HOURS), dtype=float)
    for i, (_, row) in enumerate(meta_df.iterrows()):
        pid = int(row['patient_id'])
        pdata = data_df[data_df['patient_id'] == pid].sort_values('time')
        qsofa_2d[i]  = pdata['qSOFA'].values
        labels_2d[i] = pdata['label'].values

    print("\nEvaluating …")
    results = evaluate_all(meta_df, risk_mean, labels_2d, qsofa_2d, threshold=0.5)
    print(f"  Model AUROC : {results['auroc']:.3f}")
    print(f"  qSOFA AUROC : {results['qsofa_auroc']:.3f}")
    mm = results['model_metrics']
    print(f"  Sensitivity : {mm['sensitivity']:.3f}")
    print(f"  Specificity : {mm['specificity']:.3f}")
    if results['lead_times']:
        print(f"  Median lead time vs. clinical onset: {np.median(results['lead_times']):.1f} h")

    # Save artifacts
    torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, 'model.pt'))
    joblib.dump(scaler,   os.path.join(ARTIFACT_DIR, 'scaler.pkl'))
    joblib.dump({
        'data_df':    data_df,
        'meta_df':    meta_df,
        'risk_mean':  risk_mean,
        'risk_std':   risk_std,
        'qsofa_2d':   qsofa_2d,
        'labels_2d':  labels_2d,
        'results':    results,
    }, os.path.join(ARTIFACT_DIR, 'artifacts.pkl'))

    print(f"\nArtifacts saved to {ARTIFACT_DIR}/")
    print("Launch the dashboard with:  streamlit run app/dashboard.py")


if __name__ == '__main__':
    main()
