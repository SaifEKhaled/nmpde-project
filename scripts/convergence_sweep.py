#!/usr/bin/env python3
"""
convergence_sweep.py
====================
Sweeps (scheme, refinements, fe_degree, dt) and records the final L2
error + energy ratio for every run that passes the CFL check.

Produces: results/convergence_all.csv

Scheme names accepted by the executable: CN | BE | FE | Leapfrog | RK4
  CN  = Crank-Nicolson (theta=0.5), implicit, unconditionally stable
  BE  = Backward Euler (theta=1.0), implicit, unconditionally stable
  FE  = Forward Euler  (theta=0.0), explicit, CFL-limited
  Leapfrog = Störmer-Verlet,        explicit, CFL-limited, symplectic
  RK4      = Runge-Kutta 4,         explicit, CFL-limited

CFL condition for explicit schemes on a square mesh:
    dt <= h / (c * sqrt(dim))
(We use a safety factor of 0.85 to stay clear of the limit.)

Usage (from project root):
    python3 scripts/convergence_sweep.py --nprocs 4
    python3 scripts/convergence_sweep.py --nprocs 4 --schemes CN BE
    python3 scripts/convergence_sweep.py --nprocs 4 --schemes Leapfrog RK4
"""

import subprocess, re, argparse, itertools, csv, math
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
BUILD   = ROOT / "build"
RESULTS = ROOT / "results"
EXE     = BUILD / "wave_equation"

# Explicit schemes that need the CFL check
EXPLICIT = {"FE", "Leapfrog", "RK4"}


def write_prm(path, scheme, refinements, fe_degree, dt, T=1.0, wave_speed=1.0):
    with open(path, "w") as f:
        f.write(f"set Final time    = {T}\n")
        f.write(f"set Wave speed    = {wave_speed}\n")
        f.write(f"set Refinements   = {refinements}\n")
        f.write(f"set FE degree     = {fe_degree}\n")
        f.write(f"set Scheme        = {scheme}\n")
        f.write(f"set Theta         = 0.5\n")   # ignored for non-Theta schemes
        f.write(f"set Time step     = {dt}\n")
        f.write(f"set Output every  = 0\n")     # no VTU during sweeps
        f.write(f"set Output dir    = {RESULTS}\n")


def cfl_ok(scheme, dt, refinements, fe_degree, wave_speed=1.0, safety=0.85):
    """Return False if an explicit scheme exceeds the CFL limit."""
    if scheme not in EXPLICIT:
        return True
    h      = 1.0 / (2 ** refinements)
    dt_max = h / (wave_speed * math.sqrt(2.0) * fe_degree)
    return dt <= safety * dt_max


def run_case(scheme, refinements, fe_degree, dt, nprocs, T=1.0):
    """Run one combination; return (l2_error, energy_ratio, wall_time) or None."""
    if not cfl_ok(scheme, dt, refinements, fe_degree):
        return None

    prm_path = RESULTS / "_sweep_tmp.prm"
    write_prm(prm_path, scheme, refinements, fe_degree, dt, T)

    cmd = ["mpirun", "--oversubscribe", "-np", str(nprocs), str(EXE), str(prm_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("  [TIMEOUT]")
        return None

    out = result.stdout + result.stderr
    l2     = re.search(r"Final L2 error:\s*([\d.eE+\-]+)", out)
    eratio = re.search(r"Final E/E0:\s*([\d.eE+\-]+)",     out)
    wtime  = re.search(r"Wall time:\s*([\d.eE+\-]+)",       out)

    if not l2:
        print(f"\n  [WARN] parse failed for {scheme} ref={refinements} "
              f"fe={fe_degree} dt={dt}")
        print(f"  tail: {out[-300:]}")
        return None

    # Sanity check: if energy exploded the run diverged despite passing CFL
    er_val = float(eratio.group(1)) if eratio else float("nan")
    if er_val > 100.0:
        print(f"  [DIVERGED] E/E0={er_val:.2e}")
        return None

    return (float(l2.group(1)),
            er_val,
            float(wtime.group(1)) if wtime else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nprocs",     type=int,  default=4)
    ap.add_argument("--schemes",    nargs="+",
                    default=["CN", "BE", "Leapfrog", "RK4"])
    ap.add_argument("--refs",       nargs="+", type=int,
                    default=[3, 4, 5, 6])
    ap.add_argument("--dts",        nargs="+", type=float,
                    default=[0.04, 0.02, 0.01, 0.005, 0.0025])
    ap.add_argument("--fe-degrees", nargs="+", type=int,
                    default=[1, 2])
    ap.add_argument("--T",          type=float, default=1.0)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS / "convergence_all.csv"

    rows   = []
    combos = list(itertools.product(
        args.schemes, args.refs, args.fe_degrees, args.dts))
    total  = len(combos)

    for done, (scheme, ref, fe_deg, dt) in enumerate(combos, 1):
        h   = 1.0 / (2 ** ref)
        tag = f"{scheme} ref={ref} fe={fe_deg} dt={dt:.5f}"
        print(f"[{done}/{total}] {tag} ...", end=" ", flush=True)

        if not cfl_ok(scheme, dt, ref, fe_deg):
            print("SKIP (CFL)")
            continue

        res = run_case(scheme, ref, fe_deg, dt, args.nprocs, args.T)
        if res is None:
            print("SKIP")
            continue

        l2, er, wt = res
        print(f"L2={l2:.3e}  E/E0={er:.6f}  t={wt:.1f}s")

        rows.append({
            "scheme":        scheme,
            "fe_degree":     fe_deg,
            "refinements":   ref,
            "h":             h,
            "dt":            dt,
            "dofs":          (2**ref + 1)**2,
            "l2_error":      l2,
            "energy_ratio":  er,
            "wall_time":     wt,
        })

    with open(out_csv, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nDone. {len(rows)}/{total} runs written to {out_csv}")


if __name__ == "__main__":
    main()
