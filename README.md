# Wave Equation FEM Solver — NMPDE Project

Parallel FEM solver for the 2-D wave equation:
$$u_{tt} - c^2 \Delta u = 0 \quad \text{on } (0,1)^2 \times (0,T]$$
with homogeneous Dirichlet BCs and exact standing-wave initial conditions.

Built on **deal.II** with **Trilinos** (MPI-parallel linear algebra).

---

## Project structure

```
NMPDE-Project/
├── include/
│   ├── Parameters.hpp          ← parameter struct + .prm parsing
│   ├── WaveExact.hpp           ← exact solution  u = cos(ωt) sin(πx) sin(πy)
│   ├── WaveEquationBase.hpp    ← abstract base: mesh, matrices, energy, error, output
│   ├── WaveEquationBase.tpp    ← template implementation (run loop, helpers)
│   ├── WaveTheta.hpp           ← CN / BE / FE  (theta-method)
│   ├── WaveLeapfrog.hpp        ← Störmer-Verlet (symplectic, explicit)
│   └── WaveRK4.hpp             ← 4th-order Runge-Kutta (explicit)
├── src/
│   └── main.cc                 ← reads .prm, factory-creates solver, calls run()
├── parameters/
│   ├── theta_cn.prm            ← Crank-Nicolson   (Scheme = CN)
│   ├── theta_be.prm            ← Backward Euler   (Scheme = BE)
│   ├── theta_fe.prm            ← Forward Euler    (Scheme = FE)
│   ├── leapfrog.prm            ← Störmer-Verlet   (Scheme = Leapfrog)
│   └── rk4.prm                 ← RK4              (Scheme = RK4)
├── scripts/
│   ├── convergence_sweep.py    ← spatial + temporal convergence study
│   ├── scaling_sweep.py        ← MPI strong-scaling study
│   ├── dispersion_sweep.py     ← dissipation analysis (E/E0 vs dt)
│   └── cfl_stability_sweep.py  ← CFL stability boundary for explicit schemes
├── notebooks/
│   ├── 01_Error_Analysis.ipynb      ← convergence plots + rate table
│   ├── 02_Energy_Conservation.ipynb ← dissipation analysis
│   ├── 03_CFL_Stability.ipynb       ← CFL boundary visualisation
│   └── 04_HPC_Scaling.ipynb         ← Amdahl fit + scaling plots
├── results/                    ← CSV outputs, VTU files (gitignored)
├── figures/                    ← PNG plots (gitignored)
└── CMakeLists.txt
```

---

## Schemes

| Name in .prm | Class        | Order | Stable   | Notes |
|-------------|--------------|-------|----------|-------|
| `CN`        | WaveTheta    | 2     | Unconditional | Energy-conserving |
| `BE`        | WaveTheta    | 1     | Unconditional | Dissipative |
| `FE`        | WaveTheta    | 1     | CFL: dt ≤ h/(c√2) | Explicit theta |
| `Leapfrog`  | WaveLeapfrog | 2     | CFL: dt ≤ h/(c√2) | Symplectic |
| `RK4`       | WaveRK4      | 4     | CFL: dt ≤ h/(c√2) | 4 mass solves/step |

---

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

---

## Quick test

```bash
# From project root
mpirun --oversubscribe -np 4 build/wave_equation parameters/theta_cn.prm
mpirun --oversubscribe -np 4 build/wave_equation parameters/leapfrog.prm
mpirun --oversubscribe -np 4 build/wave_equation parameters/rk4.prm
```

Each run prints:
- Scheme, DoF count, mesh size h, dt, CFL number
- Progress lines: `t = ...  E/E0 = ...  L2 = ...`
- Summary: `Final L2 error`, `Final E/E0`, `Wall time`

And writes to `results/`:
- `energy.csv`      — step, time, energy, energy_ratio
- `error.csv`       — step, time, l2_error
- `convergence.csv` — one row per run (appended)
- `wave_NNNN.*.vtu` + `.pvtu` — visualisation (if output_every > 0)

---

## Running the studies

All scripts are run from the project root.

### Convergence study (~30 min on 4 cores)
```bash
python3 scripts/convergence_sweep.py --nprocs 4
# Produces: results/convergence_all.csv
```

### Dissipation analysis
```bash
python3 scripts/dispersion_sweep.py --nprocs 4
# Produces: results/dispersion_summary.csv
```

### CFL stability boundary
```bash
python3 scripts/cfl_stability_sweep.py --nprocs 4
# Produces: results/cfl_stability.csv
```

### MPI strong scaling
```bash
python3 scripts/scaling_sweep.py --ranks 1 2 4
# Produces: results/scaling.csv
```

### Analysis notebooks
```bash
jupyter notebook notebooks/
```
Open notebooks in order (01 → 04). Each reads a CSV from `results/` and saves
plots to `results/`.

---

## CFL condition (explicit schemes)

On a uniform square mesh with Qp elements:
```
dt_CFL = h / (c * sqrt(dim) * p)
```
At ref=5 (h=1/32), c=1, p=1, dim=2:  `dt_CFL ≈ 0.022`

The convergence sweep automatically skips combinations that violate CFL
(with a 0.85 safety factor).
