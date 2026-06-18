# justfile — single entry point for building, running, and testing
#
# Install `just`: https://github.com/casey/just
#   Ubuntu/Debian: sudo apt install just
#
# Usage:
#   just              -> list all recipes
#   just build        -> configure + build (Release)
#   just ci           -> full CI mirror (build + smoke tests + lint)
#   just run-all      -> run all 5 schemes once
#   just sweeps       -> run all 5 analysis sweeps
#   just notebooks    -> launch jupyter

set shell := ["bash", "-cu"]

# Default: list recipes
default:
    @just --list

# ============================================================
# Build
# ============================================================

# Configure + build in Release mode
build:
    mkdir -p build
    cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)
    @test -x build/wave_equation && echo "OK: executable built"

# Rebuild from scratch (clean + build)
rebuild: clean build

# Remove build artifacts and stray sweep files
clean:
    rm -rf build
    rm -f results/_*.prm results/*.vtu results/*.pvtu
    @echo "Cleaned build/ and stray results/ files"

# ============================================================
# Single runs
# ============================================================

# Quick single-scheme run (default: theta_cn)
run scheme="theta_cn": build
    mkdir -p results
    mpirun --oversubscribe -np 4 build/wave_equation parameters/{{scheme}}.prm

# Run all 5 schemes once (2D)
run-all: build
    mkdir -p results
    @echo "=== CN"       && mpirun --oversubscribe -np 4 build/wave_equation parameters/theta_cn.prm
    @echo "=== BE"       && mpirun --oversubscribe -np 4 build/wave_equation parameters/theta_be.prm
    @echo "=== FE"       && mpirun --oversubscribe -np 4 build/wave_equation parameters/theta_fe.prm
    @echo "=== Leapfrog" && mpirun --oversubscribe -np 4 build/wave_equation parameters/leapfrog.prm
    @echo "=== RK4"      && mpirun --oversubscribe -np 4 build/wave_equation parameters/rk4.prm

# Run MMS verification (2D + 3D)
run-mms: build
    mkdir -p results
    @echo "=== MMS 2D" && mpirun --oversubscribe -np 4 build/wave_equation parameters/mms_cn.prm
    @echo "=== MMS 3D" && mpirun --oversubscribe -np 4 build/wave_equation parameters/mms_cn_3d.prm

# Run 3D demo
run-3d: build
    mkdir -p results
    mpirun --oversubscribe -np 4 build/wave_equation parameters/theta_cn_3d.prm

# Run with profiling enabled
run-profile: build
    mkdir -p results
    mpirun --oversubscribe -np 4 build/wave_equation parameters/profile_cn.prm

# Sweeps (analysis studies)

# Convergence sweep (~25 min, 128 runs)
sweep-convergence: build
    python3 scripts/convergence_sweep.py --nprocs 4

# Dispersion/dissipation sweep
sweep-dispersion: build
    python3 scripts/dispersion_sweep.py --nprocs 4

# CFL stability sweep
sweep-cfl: build
    python3 scripts/cfl_stability_sweep.py --nprocs 4

# MPI strong scaling sweep
sweep-scaling: build
    python3 scripts/scaling_sweep.py --ranks 1 2 4

# MPI weak scaling sweep
sweep-weak: build
    python3 scripts/weak_scaling_sweep.py --ranks 1 4 16 --refs 4 5 6
    python3 scripts/weak_scaling_sweep.py --ranks 1 4 --refs 6 7

# Run all sweeps 
sweeps: sweep-convergence sweep-dispersion sweep-cfl sweep-scaling sweep-weak
    @echo "All sweeps complete. Run 'just notebooks' to analyze."


# Launch Jupyter for the analysis notebooks
notebooks:
    jupyter notebook notebooks/


# Testing / CI
test: build
    mkdir -p results
    @echo "Crank-Nicolson (energy conservation)"
    mpirun --oversubscribe -np 2 build/wave_equation parameters/theta_cn.prm | tee /tmp/cn.log
    @python3 -c "import re; log=open('/tmp/cn.log').read(); er=float(re.search(r'Final E/E0:\s*([0-9.eE+-]+)', log).group(1)); assert abs(er-1.0)<1e-4, f'CN energy not conserved: E/E0={er}'; print(f'OK: CN E/E0 = {er}')"

    @echo "RK4 (energy + accuracy)"
    mpirun --oversubscribe -np 2 build/wave_equation parameters/rk4.prm | tee /tmp/rk4.log
    @python3 -c "import re; log=open('/tmp/rk4.log').read(); er=float(re.search(r'Final E/E0:\s*([0-9.eE+-]+)', log).group(1)); l2=float(re.search(r'Final L2 error:\s*([0-9.eE+-]+)', log).group(1)); assert abs(er-1.0)<1e-4, f'RK4 energy not conserved: E/E0={er}'; assert l2<1e-2, f'RK4 L2 too large: {l2}'; print(f'OK: RK4 E/E0={er}, L2={l2}')"

    @echo "Leapfrog (CFL stability at nominal dt)"
    mpirun --oversubscribe -np 2 build/wave_equation parameters/leapfrog.prm | tee /tmp/leapfrog.log
    @python3 -c "import re; log=open('/tmp/leapfrog.log').read(); er=float(re.search(r'Final E/E0:\s*([0-9.eE+-]+)', log).group(1)); assert 0.9<er<1.1, f'Leapfrog unstable: E/E0={er}'; print(f'OK: Leapfrog E/E0 = {er}')"

    @echo "MMS (manufactured solution, 2D)"
    mpirun --oversubscribe -np 2 build/wave_equation parameters/mms_cn.prm | tee /tmp/mms.log
    @python3 -c "import re; log=open('/tmp/mms.log').read(); l2=float(re.search(r'Final L2 error:\s*([0-9.eE+-]+)', log).group(1)); h1=float(re.search(r'Final H1 error:\s*([0-9.eE+-]+)', log).group(1)); assert l2<1e-2, f'MMS L2 too large: {l2}'; assert h1<1.0, f'MMS H1 too large: {h1}'; print(f'OK: MMS L2={l2}, H1={h1}')"

    @echo "3D standing wave (energy conservation)"
    mpirun --oversubscribe -np 2 build/wave_equation parameters/theta_cn_3d.prm | tee /tmp/cn3d.log
    @python3 -c "import re; log=open('/tmp/cn3d.log').read(); er=float(re.search(r'Final E/E0:\s*([0-9.eE+-]+)', log).group(1)); assert abs(er-1.0)<1e-4, f'3D CN energy not conserved: E/E0={er}'; print(f'OK: 3D CN E/E0 = {er}')"

    @echo ""
    @echo "All smoke tests passed."

# Check Python scripts compile and notebooks are valid JSON
lint:
    python3 -m compileall scripts/ -q && echo "OK: scripts compile"
    @for nb in notebooks/*.ipynb; do \
        python3 -c "import json; json.load(open('$nb'))" && echo "OK: $nb"; \
    done

# Full CI mirror: build + test + lint (run this before pushing)
ci: build test lint
    @echo ""
    @echo "=== CI mirror complete — safe to push"
