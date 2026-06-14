#!/usr/bin/env python3
"""
weak_scaling_sweep.py
======================
Weak scaling: problem size grows with the number of MPI ranks so that
work-per-rank stays approximately constant. Ideal weak scaling keeps
wall time constant as ranks increase.

DoFs roughly double with each refinement level in 2D (4x more cells).
To double the work per doubling of ranks, we increase `refinements` by
~0.5 per doubling (since 2D DoFs ~ 4^ref, and we want DoFs/rank constant
=> total DoFs ~ ranks => 4^delta_ref ~ ranks => delta_ref ~ log4(ranks)).

We approximate by choosing refinement levels so that DoFs/rank is as
close to constant as possible from a discrete set.

Produces: results/weak_scaling.csv

Usage:
    python3 scripts/weak_scaling_sweep.py --ranks 1 2 4
"""

import subprocess, re, argparse, csv
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


def dofs_for_ref(ref, fe_degree):
    n_per_dim = 2**ref * fe_degree + 1
    return n_per_dim ** 2


# def best_ref_for_target_dofs(target_dofs, fe_degree, ref_range=range(3, 9)):
#     """Pick the refinement level whose DoF count is closest to target."""
#     best = min(ref_range, key=lambda r: abs(dofs_for_ref(r, fe_degree) - target_dofs))
#     return best


def run_one(nprocs, prm_path):
    cmd = ["mpirun", "--oversubscribe", "-np", str(nprocs), str(EXE), str(prm_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    out = result.stdout + result.stderr
    m = re.search(r"Wall time:\s*([\d.eE+\-]+)", out)
    wall = float(m.group(1)) if m else float("nan")
    dofs_m = re.search(r"DoFs:\s*(\d+)", out)
    dofs = int(dofs_m.group(1)) if dofs_m else None
    return wall, dofs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks",      nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--scheme",     default="CN")
    ap.add_argument("--fe-degree",  type=int, default=1)
    ap.add_argument("--dt",         type=float, default=0.005)
    ap.add_argument("--T",          type=float, default=1.0)
    ap.add_argument("--refs", nargs="+", type=int, default=[4, 5, 6],
                help="Refinement level for each rank count. In 2D, "
                     "each +1 refinement gives ~4x DoFs, matching 4x ranks "
                     "for an exact weak-scaling sequence (e.g. 1,4,16 ranks "
                     "with refs 4,5,6).")
    ap.add_argument("--repeats", type=int, default=3,
                help="Repeat each run and report the minimum wall time "
                     "(reduces timing noise on small problems)")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    
    

    rows = []
    prm = RESULTS / "_weak_tmp.prm"

    if len(args.ranks) != len(args.refs):
    raise SystemExit(
        f"--ranks ({len(args.ranks)}) and --refs ({len(args.refs)}) "
        f"must have the same length"
    )

print("Weak scaling: each step should multiply both ranks and DoFs by ~4x\n")
print(f"  {'ranks':>5}  {'ref':>3}  {'DoFs':>8}  {'DoFs/rank':>10}")

for nprocs, ref in zip(args.ranks, args.refs):
    d = dofs_for_ref(ref, args.fe_degree)
    print(f"  {nprocs:>5}  {ref:>3}  {d:>8}  {d/nprocs:>10.0f}")

print()

for nprocs, ref in zip(args.ranks, args.refs):
    actual_dofs = dofs_for_ref(ref, args.fe_degree)
    dofs_per_rank = actual_dofs / nprocs

    write_prm(prm, args.scheme, ref, args.fe_degree, args.dt, args.T)

    print(f"  ranks={nprocs:>2}  ref={ref}  DoFs={actual_dofs:>7}  "
          f"DoFs/rank={dofs_per_rank:>8.0f} ...",
          end=" ", flush=True)

    times = []
    for _ in range(args.repeats):
        wall, dofs_reported = run_one(nprocs, prm)
        times.append(wall)

    best = min(times)

    print(f"runs={['%.2f'%t for t in times]} -> min={best:.2f}s")

    rows.append({
        "ranks":         nprocs,
        "refinements":   ref,
        "dofs":          dofs_reported or actual_dofs,
        "dofs_per_rank": dofs_per_rank,
        "wall_time":     best,
    })

    # Weak scaling efficiency: T(1) / T(p)  (ideal = 1.0, constant time)
    t1 = rows[0]["wall_time"]
    for r in rows:
        r["weak_efficiency"] = 100.0 * t1 / r["wall_time"]

    out = RESULTS / "weak_scaling.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\nWritten to {out}")
    print("\nWeak efficiency = T(1)/T(p) x 100%  (100% = perfect weak scaling)")
    for r in rows:
        print(f"  ranks={r['ranks']:>2}  T={r['wall_time']:.2f}s  "
              f"efficiency={r['weak_efficiency']:.1f}%")


if __name__ == "__main__":
    main()