# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 09:00:00 2026

@author: joachim.eimery
"""
import numpy as np
from numpy.linalg import svd
from scipy.linalg import eig

# ============================================================
#  DMD (Exact DMD)
# ============================================================
def dmd(X, dt, r=10):
    """
    X: (n, m) snapshots matrix
    Returns:
      Phi: DMD modes (n, r) complex
      omega: continuous-time eigenvalues (r,) complex such that exp(omega*dt)=mu
      b: mode amplitudes (r,)
    """
    X1 = X[:, :-1]
    X2 = X[:, 1:]

    # Truncated SVD of X1
    U, s, Vt = svd(X1, full_matrices=False)
    Ur = U[:, :r]
    Sr = np.diag(s[:r])
    Vr = Vt[:r, :].T

    # Low-rank operator
    Atilde = Ur.conj().T @ X2 @ Vr @ np.linalg.inv(Sr)

    mu, W = eig(Atilde)  # discrete-time eigs
    Phi = X2 @ Vr @ np.linalg.inv(Sr) @ W

    omega = np.log(mu) / dt  # continuous-time
    # Initial condition projection for amplitudes b
    x1 = X[:, 0]
    b = np.linalg.lstsq(Phi, x1, rcond=None)[0]

    return Phi, omega, b


def reconstruct_dmd(Phi, omega, b, t):
    """
    Reconstruct x(t) = sum_k b_k Phi_k exp(omega_k t)
    """
    r = len(b)
    time_dynamics = np.vstack([b[k] * np.exp(omega[k] * t) for k in range(r)])
    Xrec = Phi @ time_dynamics
    return Xrec.real