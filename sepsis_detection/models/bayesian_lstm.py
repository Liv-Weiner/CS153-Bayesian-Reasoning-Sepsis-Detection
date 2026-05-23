import torch
import torch.nn as nn
import numpy as np


class BayesianLSTM(nn.Module):
    """
    Two-layer LSTM with MC Dropout for uncertainty-aware sepsis risk scoring.

    Keeping dropout active during inference (mc_predict) turns the network into
    an approximate Bayesian model: N stochastic forward passes → empirical
    posterior mean and std over the risk score at every time step.
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)           # (batch, seq_len, hidden)
        out = self.dropout(out)
        risk = self.head(out)           # (batch, seq_len, 1)
        return risk.squeeze(-1)         # (batch, seq_len)

    def mc_predict(self, x: torch.Tensor, n_samples: int = 50):
        """
        Monte Carlo dropout inference.

        Returns
        -------
        mean : np.ndarray  shape (batch, seq_len)
        std  : np.ndarray  shape (batch, seq_len)
        """
        self.train()  # keep dropout active
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                samples.append(self.forward(x).cpu().numpy())
        self.eval()

        arr = np.stack(samples, axis=0)   # (n_samples, batch, seq_len)
        return arr.mean(axis=0), arr.std(axis=0)
