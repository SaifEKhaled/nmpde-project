# NMPDE-Wave-HPC: Matrix-Free SIMD Wave Equation Solver

![C++17](https://img.shields.io/badge/C++-17-blue.svg)
![deal.II](https://img.shields.io/badge/deal.II-%E2%89%A59.3.2-orange.svg)
![MPI](https://img.shields.io/badge/MPI-Parallel-green.svg)
![Python](https://img.shields.io/badge/Python-Dashboard-yellow.svg)

A high-performance, MPI-parallelized Finite Element Method (FEM) solver for the 2D scalar wave equation. Developed for the Numerical Mathematics and Partial Differential Equations (NMPDE) curriculum at Politecnico di Milano.

While standard FEM implementations rely on globally assembled sparse matrices and implicit time-stepping, this engine utilizes a **Matrix-Free Explicit** architecture. By trading unconditional stability for localized SIMD vectorization, this solver entirely bypasses the Von Neumann Memory Wall, achieving extreme computational throughput and super-linear parallel scaling.

---

## Architectural Highlights

### 1. Matrix-Free SIMD over Sparse Memory
Standard `TrilinosWrappers::SparseMatrix` assemblies scale at **O(N log N)** in memory and are strictly bound by RAM latency during Conjugate Gradient (CG) inversions. This engine uses a custom Matrix-Free Conjugate Gradient solver that computes the action of the mass matrix **O(N)** on-the-fly. This shifts the hardware bottleneck from memory bandwidth to the CPU's vector execution units.

### 2. Super-Linear Cache Efficiency (MPI Scaling)
Strong scaling benchmarks performed on a 20-core workstation revealed a massive **190% Parallel Efficiency** spike at 4 MPI ranks. Domain decomposition sliced the global grid into subdomains perfectly sized for the CPU's isolated L3 hardware cache, temporarily eliminating main memory latency. 

### 3. Symplectic Phase-Space Preservation
Rather than suffering from the severe numerical dissipation found in RK4 or standard Implicit solvers, this engine utilizes a Leapfrog integration scheme. Delay-embedding phase-space analysis confirms absolute conservation of Hamiltonian energy across tens of thousands of time steps.

### 4. High-Order Q4 Spectral Elements
To prevent numerical dispersion, the solver utilizes high-order quadrilateral ($Q_4$) elements rather than standard $P_1$ simplices, capturing high-frequency transient wave dynamics flawlessly.

---

## Repository Structure

```text
nmpde-project/
├── include/              # C++ headers (Matrix-Free engine, Leapfrog/RK4 integrators)
├── src/                  # C++ sources
├── parameters/           # .prm configuration files for grid and CFL control
├── scripts/              # Python benchmarking and dashboard generation tools
├── results/              # Output directory for JSON cache and ParaView .pvtu files
├── templates/            # HTML templates for the local web dashboard
└── web/                  # Python FastAPI application for dynamic data visualization
```

## Build & Execution Instructions
### Prequisites 

Compiler: C++17 compliant compiler with MPI support.
Libraries: deal.II (≥ 9.3.2) compiled with MPI and Trilinos.

## Compilation
```code
mkdir build && cd build
cmake ..
make -j
```
## Running The Engine
The application expects a parameter file to dictate the physics and resolution. Ensure safety configurations (CFL limits) are strictly obeyed when using the Explicit solver.
```code
mpirun --oversubscribe -np 4 ./wave_equation
```
## The Python Analysis Dashboard

This project decouples the heavy C++ numerical crunching from the visualization suite. The C++ engine exports telemetry to lightweight JSON caches, which are rendered instantly by a local Python dashboard.

To view the hardware scaling, phase-space topology, and comparative implicit/explicit analysis:
```code
python3 -m uvicorn web.app:app --reload
```
Navigate to http://127.0.0.1:8000 in your web browser.

## 3D Visualization (ParaView)
The engine outputs unstructured grid data compatible with ParaView for full topological rendering.
1. Open ParaView and load the `solution_..pvtu` file group from the `/results directory`.
2. Apply the Warp By Scalar filter to the displacement field.
3. Scale the Z-axis accordingly and hit Play to watch the transient standing wave propagation.
