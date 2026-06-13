#!/usr/bin/env python3
"""
scaling_sweep.py
================
MPI strong-scaling study: fixed problem size, varying number of processes.
Uses CN (Crank-Nicolson) by default — implicit, so no CFL constraint.

Produces: results/scaling.csv

Usage (from project root):
    python3 scripts/scaling_sweep.py --ranks 1 2 4
    python3 scripts/scaling_sweep.py --ranks 1 2 4 8 16  (on a cluster)
    python3 scripts/scaling_sweep.py --scheme Leapfrog --ranks 1 2 4
"""

import subprocess, re, argparse, csv, time as _time
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
BUILD   = ROOT / "build"
RESULTS = ROOT / "results"
EXE     = BUILD / "wave_equation"


def write_prm(path, scheme, refinements, fe_degree, dt, T=1.0):
    with open(path, "w") as f:
        f.write(f"set Final time    = {T}\n")
        f.write(f"set Wave speed    = 1.0\n")
        f.write(f"set Refinements   = {refinements}\n")
        f.write(f"set FE degree     = {fe_degree}\n")
        f.write(f"set Scheme        = {scheme}\n")
        f.write(f"set Theta         = 0.5\n")
        f.write(f"set Time step     = {dt}\n")
        f.write(f"set Output every  = 0\n")
        f.write(f"set Output dir    = {RESULTS}\n")


def run_one(nprocs, prm_path):
    cmd = ["mpirun", "--oversubscribe", "-np", str(nprocs), str(EXE), str(prm_path)]
    t0 = _time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    wall = _time.perf_counter() - t0

    out = result.stdout + result.stderr
    m = re.search(r"Wall time:\s*([\d.eE+\-]+)", out)
    if m:
        wall = float(m.group(1))   # prefer solver-reported time
    return wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks",       nargs="+", type=int,   default=[1, 2, 4])
    ap.add_argument("--scheme",      default="CN")
    ap.add_argument("--refinements", type=int,              default=6)
    ap.add_argument("--fe-degree",   type=int,              default=1)
    ap.add_argument("--dt",          type=float,            default=0.005)
    ap.add_argument("--T",           type=float,            default=1.0)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    prm = RESULTS / "_scaling_tmp.prm"
    write_prm(prm, args.scheme, args.refinements, args.fe_degree, args.dt, args.T)

    rows = []
    t1   = None
    for nprocs in args.ranks:
        print(f"  ranks={nprocs} ...", end=" ", flush=True)
        wall = run_one(nprocs, prm)
        if t1 is None:
            t1 = wall
        speedup    = t1 / wall
        efficiency = speedup / nprocs * 100.0
        print(f"{wall:.2f}s  speedup={speedup:.2f}x  eff={efficiency:.1f}%")
        rows.append({"ranks": nprocs, "wall_time": wall,
                     "speedup": speedup, "efficiency": efficiency})

    out = RESULTS / "scaling.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
