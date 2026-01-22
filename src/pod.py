# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 08:57:57 2026

@author: joachim.eimery
"""
import numpy as np
from numpy.linalg import svd

# ============================================================
#  POD (via SVD on centered snapshots matrix)
# ============================================================
def pod(X, r=10):
    """
    X: (n, m) snapshots, columns are time
    Returns:
      Phi: (n, r) spatial modes
      a:   (r, m) temporal coefficients
      s:   singular values (energy)
      xmean: mean field (n,)
    """
    xmean = X.mean(axis=1, keepdims=True)
    Xc = X - xmean
    U, s, Vt = svd(Xc, full_matrices=False)
    Phi = U[:, :r]
    a = (np.diag(s[:r]) @ Vt[:r, :])
    return Phi, a, s, xmean[:, 0]

