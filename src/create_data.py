# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 08:58:58 2026

@author: joachim.eimery
"""
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  Synthetic dataset: 2D scalar field u(x,y,t)
#    - convecting vortical blobs (moving Gaussians)
#    - an oscillatory standing wave component
#    - additive noise
# ============================================================
def make_synthetic_flow(nx=64, ny=64, nt=200, dt=0.05, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    U = np.zeros((nt, nx, ny), dtype=np.float64)

    # Two convecting Gaussian "vortices" (scalar proxies)
    sigma1, sigma2 = 0.15, 0.12
    vx1, vy1 = 0.35, 0.10
    vx2, vy2 = -0.25, 0.15

    # Oscillatory component: standing wave
    omega = 2.0 * np.pi * 0.8  # ~0.8 Hz (in 1/time units)
    kx, ky = 2.0 * np.pi * 1.0, 2.0 * np.pi * 0.5

    for k in range(nt):
        # print("k =", k)
        t = k * dt

        # Moving centers
        x1, y1 = -0.6 + vx1 * t, -0.2 + vy1 * t
        x2, y2 = 0.5 + vx2 * t, 0.4 + vy2 * t

        g1 = np.exp(-((X - x1) ** 2 + (Y - y1) ** 2) / (2 * sigma1**2))
        g2 = np.exp(-((X - x2) ** 2 + (Y - y2) ** 2) / (2 * sigma2**2))

        standing = np.sin(kx * X) * np.sin(ky * Y) * np.cos(omega * t)

        u = 1.0 * g1 - 0.8 * g2 + 0.6 * standing
        u += noise * rng.standard_normal((nx, ny))

        U[k] = u

    return U, dt


def flatten_snapshots(U):
    # U: (nt, nx, ny) -> X: (nspace, nt)
    nt, nx, ny = U.shape
    return U.reshape(nt, nx * ny).T


# ============================================================
#  Quick test & visualization
# ============================================================

def show_snapshot(U, k, title=""):
    print("snapshot for t =", k)
    plt.figure()
    plt.imshow(U[k], origin="lower", aspect="auto")
    plt.colorbar()
    plt.title(title or f"Snapshot t={k}")
    plt.tight_layout()
    plt.show()

