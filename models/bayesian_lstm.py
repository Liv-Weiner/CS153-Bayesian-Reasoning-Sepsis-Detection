"""
models/bayesian_lstm.py
========================
Bayesian LSTM for sequential sepsis risk scoring.

Architecture
------------
A two-layer LSTM processes ICU vital-sign sequences of arbitrary length.
Dropout layers are applied both between LSTM layers and inside the
classification head.  During inference, dropout is intentionally kept
*active* (model.train() mode) and the forward pass is repeated N times.
The resulting ensemble of predictions approximates a posterior distribution
over the risk score at each time step.

This technique — MC (Monte Carlo) Dropout — is described in:
    Gal, Y. & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning." ICML 2016.

The key insight is that a neural network trained with dropout is
mathematically equivalent to a deep Gaussian process, and running T
stochastic forward passes gives an empirical estimate of both the
posterior mean E[p(sepsis | x)] and the predictive uncertainty Var[p].
"""

import torch
import torch.nn as nn
import numpy as np


class BayesianLSTM(nn.Module):
    """
    Two-layer LSTM with MC Dropout for uncertainty-aware sepsis risk scoring.

    Input shape  : (batch, seq_len, input_size)
    Output shape : (batch, seq_len)  — sigmoid-activated risk probability
                   at every time step (sequence-to-sequence prediction)

    The model predicts risk causally: the output at time t depends only on
    inputs 0..t because the LSTM processes the sequence left-to-right without
    any look-ahead.

    Args:
        input_size  : number of vital-sign features (7 in the default pipeline)
        hidden_size : LSTM hidden dimension (default 64)
        num_layers  : number of stacked LSTM layers (default 2)
        dropout     : dropout probability applied between layers and in the
                      classification head (default 0.3)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers:  int = 2,
        dropout:   float = 0.3,
    ):
        super().__init__()
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.dropout_rate = dropout

        # LSTM backbone — inter-layer dropout is only applied when num_layers > 1
        # (PyTorch does not apply dropout after the final LSTM layer)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Additional dropout before the classification head (used during MC inference)
        self.dropout = nn.Dropout(p=dropout)

        # Two-layer MLP head: hidden → 32 → 1 (sigmoid)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Single deterministic forward pass (dropout active if model.train()).

        Args:
            x : Tensor of shape (batch, seq_len, input_size)

        Returns:
            risk : Tensor of shape (batch, seq_len) — risk probability at
                   each time step
        """
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden_size)
        lstm_out    = self.dropout(lstm_out)
        risk        = self.head(lstm_out)   # (batch, seq_len, 1)
        return risk.squeeze(-1)             # (batch, seq_len)

    def mc_predict(
        self,
        x: torch.Tensor,
        n_samples: int = 50,
    ) -> tuple:
        """
        Monte Carlo dropout inference — approximate Bayesian posterior.

        The model is set to train() mode so that dropout remains active,
        then T = n_samples stochastic forward passes are run.  The
        sample mean approximates E[p(sepsis|x)] and the sample standard
        deviation approximates the predictive uncertainty.

        Args:
            x         : Tensor of shape (batch, seq_len, input_size)
            n_samples : number of stochastic forward passes (default 50;
                        80–100 gives tighter confidence intervals)

        Returns:
            mean : np.ndarray of shape (batch, seq_len) — posterior mean risk
            std  : np.ndarray of shape (batch, seq_len) — posterior std (used
                   to construct the 95 % credible interval as mean ± 1.96·std)
        """
        self.train()  # activate dropout for stochastic sampling
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                samples.append(self.forward(x).cpu().numpy())
        self.eval()   # restore deterministic mode after sampling

        arr = np.stack(samples, axis=0)  # (n_samples, batch, seq_len)
        return arr.mean(axis=0), arr.std(axis=0)
