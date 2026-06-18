#!/usr/bin/env python3
"""
cfl_stability_sweep.py
======================
Demonstrates the CFL stability boundary for explicit schemes
(Leapfrog and RK4) by running at dt = fraction * dt_CFL for
fractions in [0.5, 0.75, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20].

Implicit schemes (CN, BE) are unconditionally stable and are not tested.

Theoretical CFL limit on a uniform square mesh with Q1 elements:
    dt_CFL = h / (c * sqrt(dim))   where dim=2

We use a short T to quickly see if the run diverges (E/E0 >> 1).

Produces: results/cfl_stability.csv

Usage:
    python3 scripts/cfl_stability_sweep.py --nprocs 4
"""

import subprocess, re, argparse, csv, math
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
BUILD   = ROOT / "build"
RESULTS = ROOT / "results"
EXE     = BUILD / "wave_equation"

REFS      = 5
T_FINAL   = 2.0    # long enough for instability to grow visibly


def dt_cfl(wave_speed=1.0):
    h = 1.0 / (2 ** REFS)
    return h / (wave_speed * math.sqrt(2.0))


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


def run(scheme, dt, nprocs):
    prm = RESULTS / "_cfl_tmp.prm"
    write_prm(prm, scheme, dt)
    cmd = ["mpirun", "--oversubscribe", "-np", str(nprocs), str(EXE), str(prm)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    out = r.stdout + r.stderr
    er  = re.search(r"Final E/E0:\s*([\d.eE+\-]+)", out)
    if not er:
        return None, "CRASH"
    val = float(er.group(1))
    if val > 1e4:
        status = "UNSTABLE"
    elif val > 5.0:
        status = "DIVERGING"
    elif val > 1.5:
        status = "GROWING"
    else:
        status = "STABLE"
    return val, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nprocs",    type=int,  default=4)
    ap.add_argument("--schemes",   nargs="+", default=["Leapfrog", "RK4"])
    ap.add_argument("--fractions", nargs="+", type=float,
                    default=[0.5, 0.75, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.40, 1.60, 2.00])
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    dtc = dt_cfl()
    h   = 1.0 / (2 ** REFS)
    print(f"Mesh: ref={REFS}  h={h:.5f}  dt_CFL = {dtc:.6f}\n")

    rows = []
    for scheme in args.schemes:
        print(f"── {scheme} ──")
        for frac in args.fractions:
            dt = frac * dtc
            er, status = run(scheme, dt, args.nprocs)
            er_str = f"{er:.4e}" if er is not None else "N/A"
            print(f"  dt/dt_CFL={frac:.2f}  dt={dt:.6f}  "
                  f"E/E0={er_str:>14}  [{status}]")
            rows.append({
                "scheme":       scheme,
                "dt_fraction":  frac,
                "dt":           dt,
                "energy_ratio": er if er is not None else float("nan"),
                "status":       status,
            })
        print()

    out = RESULTS / "cfl_stability.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Written to {out}")


if __name__ == "__main__":
    main()