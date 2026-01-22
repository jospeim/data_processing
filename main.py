# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 08:37:21 2026

@author: joachim.eimery
"""
import numpy as np
import matplotlib.pyplot as plt
from src.logger import create_result_file, write_result
from src.create_data import make_synthetic_flow, flatten_snapshots, show_snapshot
from src.pod import pod
from src.spod import spod
from src.DMD import reconstruct_dmd, dmd
# from src.AutoEncoder import train_autoencoder, ae_reconstruct

def main():
    result_file = create_result_file()
    write_result(result_file, "start script")
    # --- Generate dataset
    U, dt = make_synthetic_flow(nx=64, ny=64, nt=240, dt=0.05, noise=0.02, seed=1)
    nt, nx, ny = U.shape
    X = flatten_snapshots(U)  # (nspace, nt)
    # nspace = X.shape[0]
    write_result(result_file, "end of generation")
    show_snapshot(U, 0, "True field at t0")
    show_snapshot(U, nt // 2, "True field at mid-time")
    
    
    # --- POD
    write_result(result_file, "start pod")
    r_pod = 8
    Phi_pod, a_pod, s_pod, xmean = pod(X, r=r_pod)
    write_result(result_file, "end pod")
    # POD reconstruction using first r modes
    Xc = X - xmean[:, None]
    X_pod = (Phi_pod @ (Phi_pod.T @ Xc)) + xmean[:, None]
    U_pod = X_pod.T.reshape(nt, nx, ny)
    show_snapshot(U_pod, nt // 2, f"POD recon (r={r_pod}) at mid-time")

    # --- SPOD
    # pick relatively small nfft for speed; increase for better freq resolution
    write_result(result_file, "start spod")
    freqs, modes_spod, evals_spod = spod(X, dt=dt, nfft=64, overlap=0.5, r=3)
    # show leading eigenvalue spectrum (first mode) vs frequency
    write_result(result_file, "end spod")

    lam1 = np.array([ev[0] for ev in evals_spod])
    plt.figure()
    plt.plot(freqs, lam1)
    plt.xlabel("Frequency")
    plt.ylabel("Leading SPOD eigenvalue")
    plt.title("SPOD: leading mode energy vs frequency")
    plt.tight_layout()

    # visualize one SPOD mode at a chosen frequency bin (e.g., closest to 0.8)
    f_target = 0.8
    fi = int(np.argmin(np.abs(freqs - f_target)))
    mode0 = modes_spod[fi][:, 0].real.reshape(nx, ny)
    plt.figure()
    plt.imshow(mode0, origin="lower", aspect="auto")
    plt.colorbar()
    plt.title(f"SPOD mode 1 (real part) at f≈{freqs[fi]:.3f}")
    plt.tight_layout()

    # --- DMD
    r_dmd = 10
    Phi_dmd, omega, b = dmd(X, dt=dt, r=r_dmd)
    # reconstruct full time window
    t = np.arange(nt) * dt
    X_dmd = reconstruct_dmd(Phi_dmd, omega, b, t)
    U_dmd = X_dmd.T.reshape(nt, nx, ny)
    show_snapshot(U_dmd, nt // 2, f"DMD recon (r={r_dmd}) at mid-time")

    # Show DMD frequencies (imag part / 2π) and growth rates (real)
    freqs_dmd = omega.imag / (2 * np.pi)
    growth = omega.real
    plt.figure()
    plt.scatter(freqs_dmd, growth)
    plt.xlabel("DMD frequency (Hz-equivalent)")
    plt.ylabel("Growth rate (real(omega))")
    plt.title("DMD spectrum")
    plt.tight_layout()

    # --- Autoencoder
    # Use a smaller subset for speed if needed
    # r_ae = 8
    # model, Z = train_autoencoder(X, n_latent=r_ae, epochs=30, batch_size=32, lr=1e-3)
    # X_ae = ae_reconstruct(model, X)
    # U_ae = X_ae.T.reshape(nt, nx, ny)
    # show_snapshot(U_ae, nt // 2, f"AE recon (latent={r_ae}) at mid-time")

    # --- Compare reconstruction error (relative)
    def rel_err(Xhat):
        return np.linalg.norm(X - Xhat) / (np.linalg.norm(X) + 1e-12)

    print("\nReconstruction relative errors:")
    print(f"  POD (r={r_pod}): {rel_err(X_pod):.4f}")
    print(f"  DMD (r={r_dmd}): {rel_err(X_dmd):.4f}")
    # print(f"  AE  (z={r_ae}) : {rel_err(X_ae):.4f}")

    plt.show()


if __name__ == "__main__":
    main()
