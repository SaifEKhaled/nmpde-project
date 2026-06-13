# Wave Equation FEM Solver — NMPDE Project

Parallel finite element solver for the 2-D acoustic wave equation:

$$u_{tt} - c^2 \Delta u = 0 \quad \text{on } (0,1)^2 \times (0,T]$$

with homogeneous Dirichlet BCs and standing-wave initial conditions
$u_0 = \sin(\pi x)\sin(\pi y)$, $v_0 = 0$.

Built on **deal.II 9.3** with **Trilinos** (MPI-parallel sparse linear algebra).
Five time-integration schemes are implemented, studied, and compared.

---

## Results at a glance

### Spatial convergence (Q1 and Q2 elements)

| Scheme | Space Q1 | Observed | Space Q2 | Observed |
|--------|----------|----------|----------|----------|
| CN (Newmark β=0.25) | O(h²) | **2.03** ✓ | O(h³) | ~2.0† |
| BE (Newmark β=0.50) | O(h²) | **2.09** ✓ | O(h³) | ~2.0† |
| Leapfrog | O(h²) | ~0.65‡ | O(h³) | ~0‡ |
| RK4 | O(h²) | **1.99** ✓ | O(h³) | **3.08** ✓ |

† CN/BE Q2 spatial error O(h³) falls below temporal error O(dt²) at fine meshes;
rate measured only at h=0.125 where spatial dominates.  
‡ CFL constraint forces dt ∝ h, coupling spatial and temporal errors.

### Energy conservation at T=1, ref=5, Q1

| Scheme | E(T)/E(0) | L2 error | Wall time |
|--------|-----------|----------|-----------|
| CN | **1.000000** | 7.24e-04 | 1.15s |
| BE | 1.000459 | 2.04e-04 | 1.18s |
| Leapfrog | 0.9806 | 2.27e-02 | **0.44s** |
| RK4 | **1.000000** | 1.07e-03 | 2.06s |

### MPI strong scaling (CN, ref=6, 4 cores)

| Ranks | Wall time | Speedup | Efficiency |
|-------|-----------|---------|------------|
| 1 | 23.68s | 1.00× | 100% |
| 2 | 13.21s | 1.79× | **89.7%** |
| 4 | 8.73s | 2.71× | 67.9% |

Amdahl serial fraction: **14.5%** → theoretical max speedup 6.9×.

### CFL stability (Leapfrog, ref=5)

| dt / dt_CFL | E(T)/E(0) | Status |
|-------------|-----------|--------|
| 0.95 | 1.089 | STABLE |
| 1.00 | 6.6×10⁵⁴ | **UNSTABLE** |
| 1.05 | 1.098 | STABLE* |
| 1.40 | 1.18×10¹¹³ | UNSTABLE |

*Near-boundary behaviour reflects the discrete FEM spectrum, not a simple scalar CFL limit.

---

## Schemes

| `.prm` name | Formulation | Time order | Stability | Energy |
|-------------|-------------|-----------|-----------|--------|
| `CN` | Newmark β=0.25, γ=0.5 | 2nd | Unconditional | Conserving |
| `BE` | Newmark β=0.50, γ=0.5 | 2nd | Unconditional | Near-conserving |
| `FE` | Newmark β=0.00, γ=0.5 | 1st | CFL: dt ≤ h/(c√2) | Explicit |
| `Leapfrog` | Störmer-Verlet | 2nd | CFL: dt ≤ h/(c√2) | Symplectic |
| `RK4` | Classical RK4 | 4th | CFL: dt ≤ h/(c√2) | Near-conserving |

**CN** uses the Newmark average-acceleration method (β=0.25, γ=0.5), which
conserves the discrete energy $E_h = \frac{1}{2}(v^T M v + c^2 u^T K u)$ exactly.
Note: the first-order system CN splitting does *not* conserve this energy — the
Newmark displacement formulation is required.

---

## Project structure

```
├── include/
│   ├── Parameters.hpp          ← parameter struct + .prm parsing
│   ├── WaveExact.hpp           ← exact solution + ExactVelocity for BCs
│   ├── WaveEquationBase.hpp    ← abstract base: mesh, FE, matrices, energy, error
│   ├── WaveEquationBase.tpp    ← run loop, output, convergence logging
│   ├── WaveTheta.hpp           ← Newmark-beta (CN / BE / FE)
│   ├── WaveLeapfrog.hpp        ← Störmer-Verlet
│   └── WaveRK4.hpp             ← RK4
├── src/main.cc                 ← factory: reads .prm, creates solver, calls run()
├── parameters/                 ← one .prm file per scheme
├── scripts/
│   ├── convergence_sweep.py    ← 160-run sweep over (scheme, ref, fe_degree, dt)
│   ├── dispersion_sweep.py     ← E(T)/E(0) vs dt for all schemes
│   ├── cfl_stability_sweep.py  ← stability boundary for explicit schemes
│   └── scaling_sweep.py        ← MPI strong-scaling study
├── notebooks/
│   ├── 01_Error_Analysis.ipynb      ← spatial/temporal convergence plots + rate table
│   ├── 02_Energy_Conservation.ipynb ← dissipation analysis
│   ├── 03_CFL_Stability.ipynb       ← CFL boundary plot
│   └── 04_HPC_Scaling.ipynb         ← Amdahl fit + speedup/efficiency plots
└── CMakeLists.txt
```

---

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Requires deal.II ≥ 9.3 with Trilinos and MPI. If deal.II is not on the default
path: `cmake -DDEAL_II_DIR=/path/to/dealii ..`

---

## Quick start

```bash
# From project root — create results dir if it doesn't exist
mkdir -p results

mpirun --oversubscribe -np 4 build/wave_equation parameters/theta_cn.prm
mpirun --oversubscribe -np 4 build/wave_equation parameters/theta_be.prm
mpirun --oversubscribe -np 4 build/wave_equation parameters/leapfrog.prm
mpirun --oversubscribe -np 4 build/wave_equation parameters/rk4.prm
```

Each run prints a header (scheme, DoFs, h, dt, CFL), live progress
(`t = ...  E/E0 = ...  L2 = ...`), and a final summary line. Output files
written to `results/`: `energy.csv`, `error.csv`, `convergence.csv`, and
`.vtu` / `.pvtu` files for ParaView.

---

## Running the full study

```bash
# Convergence sweep — 128 runs, ~25 min on 4 cores
python3 scripts/convergence_sweep.py --nprocs 4

# Dissipation analysis
python3 scripts/dispersion_sweep.py --nprocs 4

# CFL stability boundary
python3 scripts/cfl_stability_sweep.py --nprocs 4

# MPI strong scaling
python3 scripts/scaling_sweep.py --ranks 1 2 4

# Analysis notebooks
jupyter notebook notebooks/
```

The convergence sweep automatically skips CFL-violating combinations for
explicit schemes (safety factor 0.85). It also discards any run where
E/E0 > 100 (diverged despite passing the CFL check).

---

## Implementation notes

**Energy conservation fix**: an earlier version using the first-order
system formulation of CN (velocity-acceleration split) showed E/E0 ≈ 0.90
at T=1. This is a known property of that splitting — it does not conserve
$\frac{1}{2}(v^2 + \omega^2 u^2)$. The solver was rewritten to use the
**Newmark-beta displacement formulation**, which provably conserves discrete
energy for β=0.25, γ=0.5.

**Ghosted vector restriction**: deal.II Trilinos vectors initialised with
both `locally_owned_dofs` and `locally_relevant_dofs` (ghosted) are
read-only. All arithmetic is performed on owned vectors; ghosted copies
are created only for `DataOut` output.

**Scheme factory**: `main.cc` is 50 lines. Adding a new scheme requires
one new `.hpp` file and one `else if` in the factory — the base class
handles all mesh setup, matrix assembly, energy tracking, VTU output,
and convergence logging.