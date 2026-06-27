#!/usr/bin/env python
"""
Compare NN-RKPM backends (TensorFlow / PyTorch / JAX) for accuracy and speed.

Every backend writes the *same* artifacts to its own ``--out-dir``:
  * ``results_nnrk_LBFGS.txt`` -- per-cell ``[x y z | ux uy uz | uy_nn | e11 e22 e33 g23 g13 g12]``
  * ``loss_lbfgs.txt``         -- L-BFGS loss history (last entry = converged energy)
  * ``timings_<backend>.txt``  -- ``[adam_rk, adam_nnrk, lbfgs, total]`` seconds
so comparison is just loading and diffing -- no retraining.

  * Accuracy: converged total potential energy, and per-field max/mean |diff| vs a reference
    backend (they solve the same BVP, so the converged fields should agree to a few sig figs;
    they are not bit-identical -- each framework's RNG seeds the enrichment MLP differently).
  * Speed: per-stage and total wall time.

Usage:
  python compare_backends.py                      # default torch + jax out-dirs under --input-dir
  python compare_backends.py torch:DIR jax:DIR [tf:DIR] [--ref torch]
"""
import argparse
import os

import numpy as np

COLS = ["x", "y", "z", "ux", "uy", "uz", "uy_nn", "e11", "e22", "e33", "g23", "g13", "g12"]
FIELDS = COLS[3:]  # skip the identical xyz coordinates


def load_backend(label, out_dir):
    res = os.path.join(out_dir, "results_nnrk_LBFGS.txt")
    if not os.path.exists(res):
        return None
    b = {"label": label, "dir": out_dir, "results": np.loadtxt(res)}
    loss = os.path.join(out_dir, "loss_lbfgs.txt")
    b["final_energy"] = float(np.loadtxt(loss).reshape(-1)[-1]) if os.path.exists(loss) else float("nan")
    tf = os.path.join(out_dir, f"timings_{label}.txt")
    b["timings"] = np.loadtxt(tf).reshape(-1) if os.path.exists(tf) else None
    return b


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("specs", nargs="*", help="label:out_dir entries (e.g. torch:/path jax:/path)")
    ap.add_argument("--input-dir", default="Input_3Dbimat_Node396_Cell2662")
    ap.add_argument("--ref", default=None, help="reference label for field diffs (default: first)")
    args = ap.parse_args()

    if args.specs:
        specs = [s.split(":", 1) for s in args.specs]
    else:
        specs = [["torch", os.path.join(args.input_dir, "Results_40NR_4hidden_5bases")],
                 ["jax", os.path.join(args.input_dir, "Results_40NR_4hidden_5bases_jax")]]

    backends = [b for b in (load_backend(lbl, d) for lbl, d in specs) if b is not None]
    if not backends:
        raise SystemExit("no backends found (looked for results_nnrk_LBFGS.txt in each --out-dir)")
    missing = [f"{lbl} ({d})" for lbl, d in specs if not os.path.exists(os.path.join(d, "results_nnrk_LBFGS.txt"))]
    if missing:
        print("WARNING: skipped (no results found):", ", ".join(missing))

    # ---- speed + converged energy ----
    print("\n=== converged energy & wall time ===")
    print(f"{'backend':8s} {'final energy':>16s} {'adam_rk':>10s} {'adam_nnrk':>11s} {'lbfgs':>10s} {'total':>10s}")
    for b in backends:
        t = b["timings"]
        ts = (f"{t[0]:10.1f} {t[1]:11.1f} {t[2]:10.1f} {t[3]:10.1f}" if t is not None
              else f"{'n/a':>10s} {'n/a':>11s} {'n/a':>10s} {'n/a':>10s}")
        print(f"{b['label']:8s} {b['final_energy']:16.6e} {ts}")

    # ---- accuracy: per-field diff vs reference ----
    ref_label = args.ref or backends[0]["label"]
    ref = next((b for b in backends if b["label"] == ref_label), backends[0])
    others = [b for b in backends if b is not ref]
    if not others:
        print("\n(only one backend present -- nothing to diff)")
        return

    print(f"\n=== field agreement vs '{ref['label']}'  (max | mean abs diff over {ref['results'].shape[0]} cells) ===")
    hdr = "field   " + "".join(f"{b['label']:>22s}" for b in others)
    print(hdr)
    R = ref["results"]
    for k, name in enumerate(COLS):
        if name not in FIELDS:
            continue
        cells = []
        for b in others:
            if b["results"].shape != R.shape:
                cells.append(f"{'shape mismatch':>22s}")
                continue
            d = np.abs(b["results"][:, k] - R[:, k])
            cells.append(f"{d.max():>10.2e} {d.mean():>10.2e} ")
        print(f"{name:7s} " + "".join(cells))
    print("\n(displacements/strains agreeing to ~1e-3..1e-6 across backends is expected: same BVP,"
          "\n different RNG-seeded enrichment init; the converged energies should match to a few sig figs.)")


if __name__ == "__main__":
    main()
