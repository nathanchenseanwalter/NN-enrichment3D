#!/usr/bin/env python
"""
Plot the converged NN-RKPM results saved by ``nnpu_torch.py``.

Reads ``results_nnrk_LBFGS.txt`` (per-cell coordinates, displacements, NN part and
strains), recomputes the full stress tensor + von Mises from the bimaterial moduli,
and reads ``Enrichment_cell_LBFGS.txt`` for the learned basis. Produces multi-panel
figures (z = 0 mid-plane slices, plus a 3D view and a von-Mises z-montage) and writes
them back into the results directory. No retraining required.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def slice_mask(X, z_target=0.0):
    zvals = np.unique(np.round(X[:, 2], 6))
    z0 = zvals[np.argmin(np.abs(zvals - z_target))]
    return np.abs(X[:, 2] - z0) < 1e-4, float(z0)


def panel(ax, X, v, mask, title, cmap="jet"):
    sc = ax.tripcolor(X[mask, 0], X[mask, 1], np.asarray(v)[mask], cmap=cmap, shading="gouraud")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(sc, ax=ax, shrink=0.85)


def grid_figure(X, mask, items, fname, suptitle, ncols=3):
    n = len(items)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows), squeeze=False)
    for k, (title, v, cmap) in enumerate(items):
        panel(axes[k // ncols][k % ncols], X, v, mask, title, cmap)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input-dir", default="Input_3Dbimat_Node396_Cell2662")
    ap.add_argument("--out-dir", default=None, help="default: <input-dir>/Results_40NR_4hidden_5bases")
    ap.add_argument("--results", default="results_nnrk_LBFGS.txt")
    ap.add_argument("--enrichment", default="Enrichment_cell_LBFGS.txt")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.input_dir, "Results_40NR_4hidden_5bases")
    r = np.loadtxt(os.path.join(out_dir, args.results))            # (n_cell, 13)
    X = r[:, 0:3]
    ux, uy, uz, uy_nn = r[:, 3], r[:, 4], r[:, 5], r[:, 6]
    e11, e22, e33, g23, g13, g12 = (r[:, 7], r[:, 8], r[:, 9], r[:, 10], r[:, 11], r[:, 12])
    umag = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)

    # bimaterial moduli (matches nnpu_torch.py): matrix label 0, inclusion otherwise
    Vlabel = np.loadtxt(os.path.join(args.input_dir, "Vlabel.dat")).reshape(-1)
    E = np.where(Vlabel == 0, 10400.0, 52000.0)
    nu = 0.3
    mu = E / 2.0 / (1.0 + nu)
    lam = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu)
    M = 2.0 * mu + lam
    s11 = M * e11 + lam * (e22 + e33)
    s22 = M * e22 + lam * (e11 + e33)
    s33 = M * e33 + lam * (e11 + e22)
    s12, s23, s13 = mu * g12, mu * g23, mu * g13
    vm = np.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2)
                 + 3.0 * (s12 ** 2 + s23 ** 2 + s13 ** 2))

    mask, z0 = slice_mask(X, 0.0)
    tag = f"(z = {z0:g} slice, {mask.sum()} cells)"

    grid_figure(X, mask, [
        ("ux", ux, "jet"), ("uy", uy, "jet"), ("uz", uz, "jet"),
        ("|u|", umag, "viridis"), ("uy (NN enrichment part)", uy_nn, "coolwarm"),
        ("material label (0=matrix, 1=inclusion)", Vlabel, "coolwarm"),
    ], os.path.join(out_dir, "results_displacement.png"), f"Displacements {tag}")

    grid_figure(X, mask, [
        ("e11", e11, "jet"), ("e22", e22, "jet"), ("e33", e33, "jet"),
        ("g12", g12, "jet"), ("g23", g23, "jet"), ("g13", g13, "jet"),
    ], os.path.join(out_dir, "results_strain.png"), f"Strains {tag}")

    grid_figure(X, mask, [
        ("s11 [MPa]", s11, "jet"), ("s22 [MPa]", s22, "jet"), ("s33 [MPa]", s33, "jet"),
        ("s12 [MPa]", s12, "jet"), ("von Mises [MPa]", vm, "inferno"),
        ("material label (0=matrix, 1=inclusion)", Vlabel, "coolwarm"),
    ], os.path.join(out_dir, "results_stress.png"), f"Stresses {tag}")

    enr_path = os.path.join(out_dir, args.enrichment)
    if os.path.exists(enr_path):
        zeta = np.loadtxt(enr_path)
        items = [(f"enrichment basis {k}", zeta[:, k], "jet") for k in range(zeta.shape[1])]
        items.append(("material label (0=matrix, 1=inclusion)", Vlabel, "coolwarm"))
        grid_figure(X, mask, items, os.path.join(out_dir, "results_enrichment.png"),
                    f"NN enrichment basis {tag}")

    # von Mises across all z-planes
    zvals = np.unique(np.round(X[:, 2], 6))
    ncols = 4
    nrows = (len(zvals) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.6 * nrows), squeeze=False)
    vmin, vmax = np.percentile(vm, 1), np.percentile(vm, 99)
    for k, z in enumerate(zvals):
        m = np.abs(X[:, 2] - z) < 1e-4
        ax = axes[k // ncols][k % ncols]
        sc = ax.tripcolor(X[m, 0], X[m, 1], vm[m], cmap="inferno", shading="gouraud", vmin=vmin, vmax=vmax)
        ax.set_title(f"z = {z:g}", fontsize=10); ax.set_aspect("equal")
    for k in range(len(zvals), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.colorbar(sc, ax=axes, shrink=0.6, label="von Mises [MPa]")
    fig.suptitle("von Mises stress across z-planes", fontsize=13)
    fig.savefig(os.path.join(out_dir, "results_vonmises_zslices.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(out_dir, "results_vonmises_zslices.png"))

    # 3D scatter: uy and von Mises
    fig = plt.figure(figsize=(11, 5))
    for i, (v, label, cmap) in enumerate([(uy, "uy", "jet"), (vm, "von Mises [MPa]", "inferno")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        p = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=v, cmap=cmap, s=6, alpha=0.35)
        ax.set_title(label); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_box_aspect([1, 2, 1])
        fig.colorbar(p, ax=ax, shrink=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "results_3d.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(out_dir, "results_3d.png"))

    print(f"\nsummary: |u|max={umag.max():.4e}  uy in [{uy.min():.3e},{uy.max():.3e}]  "
          f"vonMises in [{vm.min():.1f},{vm.max():.1f}] MPa")


if __name__ == "__main__":
    main()
