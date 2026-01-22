# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 08:59:51 2026

@author: joachim.eimery
"""
import numpy as np

# ============================================================
#  SPOD (minimal Welch-style): DFT over blocks + eig of CSD
#    This is a simple reference implementation for learning/testing.
# ============================================================
def spod(X, dt, nfft=64, overlap=0.5, window="hann", r=3, detrend_mean=True, verbose=False):
    """
    SPOD via Welch: build Q(ω) (nspace x nblocks) for each ω and compute SVD(Q).
    X: (nspace, nt) snapshots matrix (columns are time)
    Returns:
      freqs: (nfreq,)
      modes: list length nfreq, each (nspace, r) complex SPOD modes
      evals: list length nfreq, each (r,) eigenvalues (energies) per frequency
    """
    n, m = X.shape
    step = int(nfft * (1 - overlap))
    if step < 1:
        raise ValueError("overlap too large (step < 1)")

    # Window
    if window == "hann":
        w = np.hanning(nfft)
    elif window == "hamming":
        w = np.hamming(nfft)
    else:
        w = np.ones(nfft)

    # Normalize window energy (common convention)
    w = w / np.sqrt(np.mean(w**2))

    # Segment indices
    starts = np.arange(0, m - nfft + 1, step)
    nblk = len(starts)
    if nblk < 2:
        raise ValueError("Not enough blocks; increase nt or reduce nfft/overlap")

    # Remove temporal mean if requested
    Xc = X - X.mean(axis=1, keepdims=True) if detrend_mean else X

    # FFT frequencies
    freqs = np.fft.rfftfreq(nfft, d=dt)
    nfreq = len(freqs)

    # ------------------------------------------------------------
    # Key optimization:
    # Precompute block FFTs once:
    # Fblocks shape: (nblk, nspace, nfreq)
    # ------------------------------------------------------------
    Fblocks = np.empty((nblk, n, nfreq), dtype=np.complex128)

    for bi, s0 in enumerate(starts):
        seg = Xc[:, s0:s0 + nfft]          # (nspace, nfft)
        segw = seg * w[None, :]            # apply window
        Fblocks[bi] = np.fft.rfft(segw, axis=1)  # (nspace, nfreq)

    # For each frequency, build Q = [q1 ... q_nblk] and SVD
    modes = []
    evals = []

    for fi in range(nfreq):
        if verbose:
            print("fi =", fi)

        # Q: (nspace, nblk) from precomputed FFT blocks
        Q = Fblocks[:, :, fi].T  # (nspace, nblk)

        # SVD(Q): Q = U Σ V*
        # SPOD modes = U, eigenvalues = Σ^2 / nblk
        Uq, sq, _ = np.linalg.svd(Q, full_matrices=False)
        modes.append(Uq[:, :r])
        evals.append((sq[:r]**2) / nblk)

    return freqs, modes, evals
