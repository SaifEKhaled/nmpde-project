#!/usr/bin/env python3
"""
dispersion_sweep.py
===================
Studies numerical dissipation of each scheme by running at a range of
dt values and recording E(T)/E(0).

Key signature:
  CN        (theta=0.5): E/E0 ≈ 1.000  (no dissipation, energy-conserving)
  BE        (theta=1.0): E/E0 < 1      (dissipative — damps high frequencies)
  Leapfrog:              E/E0 ≈ 1      (symplectic — conserves modified energy)
  RK4:                   E/E0 ≈ 1      (tiny dissipation, O(dt^4)*T)

Uses a fine fixed mesh so spatial error is negligible; varies only dt.

Produces: results/dispersion_summary.csv

Usage (from project root):
    python3 scripts/dispersion_sweep.py --nprocs 4
    python3 scripts/dispersion_sweep.py --nprocs 4 --schemes CN BE
"""

import subprocess, re, argparse, csv, math
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
BUILD   = ROOT / "build"
RESULTS = ROOT / "results"
EXE     = BUILD / "wave_equation"

EXPLICIT  = {"FE", "Leapfrog", "RK4"}
# Fine mesh — spatial error ~O(h^2) = O(1/1024) is negligible
REFS      = 5
T_FINAL   = 2.0    # run longer so dissipation is visible


def write_prm(path, scheme, dt):
    with open(path, "w") as f:
        f.write(f"set Final time    = {T_FINAL}\n")
        f.write(f"set Wave speed    = 1.0\n")
        f.write(f"set Refinements   = {REFS}\n")
        f.write(f"set FE degree     = 1\n")
        f.write(f"set Scheme        = {scheme}\n")
        f.write(f"set Theta         = 0.5\n")
        f.write(f"set Time step     = {dt}\n")
        f.write(f"set Output every  = 0\n")
        f.write(f"set Output dir    = {RESULTS}\n")


def cfl_ok(scheme, dt, safety=0.85):
    h      = 1.0 / (2 ** REFS)
    dt_max = h / math.sqrt(2.0)
    return scheme not in EXPLICIT or dt <= safety * dt_max


def run(scheme, dt, nprocs):
    if not cfl_ok(scheme, dt):
        return None
    prm = RESULTS / "_disp_tmp.prm"
    write_prm(prm, scheme, dt)
    cmd = ["mpirun", "--oversubscribe", "-np", str(nprocs), str(EXE), str(prm)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return None
    out = r.stdout + r.stderr
    er  = re.search(r"Final E/E0:\s*([\d.eE+\-]+)",     out)
    l2  = re.search(r"Final L2 error:\s*([\d.eE+\-]+)", out)
    if not er:
        return None
    er_val = float(er.group(1))
    if er_val > 100.0:   # diverged
        return None
    return er_val, float(l2.group(1)) if l2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nprocs",  type=int,  default=4)
    ap.add_argument("--schemes", nargs="+",
                    default=["CN", "BE", "Leapfrog", "RK4"])
    ap.add_argument("--dts",     nargs="+", type=float,
                    default=[0.02, 0.01, 0.005, 0.002])
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []

    for scheme in args.schemes:
        for dt in args.dts:
            print(f"  {scheme} dt={dt} ...", end=" ", flush=True)
            res = run(scheme, dt, args.nprocs)
            if res is None:
                print("SKIP")
                continue
            er, l2 = res
            print(f"E/E0={er:.6f}  L2={l2:.3e}")
            rows.append({"scheme": scheme, "dt": dt,
                         "energy_ratio": er, "l2_error": l2})

    out = RESULTS / "dispersion_summary.csv"
    with open(out, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
