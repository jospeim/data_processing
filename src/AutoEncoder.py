# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 09:00:11 2026

@author: joachim.eimery
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# ============================================================
#  Autoencoder (minimal fully-connected)
# ============================================================
class AE(nn.Module):
    def __init__(self, n_in, n_latent=8, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_in),
        )

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat, z


def train_autoencoder(X, n_latent=8, epochs=30, batch_size=32, lr=1e-3, device=None):
    """
    X: (n, m) snapshots matrix
    Train on columns as samples: m samples of dimension n
    Returns model + latent codes Z: (n_latent, m)
    """
    n, m = X.shape
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Standardize for NN training
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    Xn = (X - mean) / std

    data = torch.tensor(Xn.T, dtype=torch.float32)  # (m, n)
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

    model = AE(n_in=n, n_latent=n_latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        total = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            xhat, _ = model(xb)
            loss = loss_fn(xhat, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        # print minimal info
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"[AE] epoch {ep+1:03d}/{epochs}  loss={total/m:.6f}")

    # Encode all
    model.eval()
    with torch.no_grad():
        Z = model.encoder(data.to(device)).cpu().numpy().T  # (n_latent, m)

    # Store normalization params for reconstruction usage
    model._x_mean = mean
    model._x_std = std
    return model, Z


def ae_reconstruct(model, X):
    n, m = X.shape
    mean = model._x_mean
    std = model._x_std
    Xn = (X - mean) / std
    device = next(model.parameters()).device
    with torch.no_grad():
        data = torch.tensor(Xn.T, dtype=torch.float32).to(device)
        xhat, _ = model(data)
        Xhat = xhat.cpu().numpy().T
    return Xhat * std + mean
