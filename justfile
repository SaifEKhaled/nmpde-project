# justfile — run the same checks locally that CI runs
#
# Install `just`: https://github.com/casey/just
#   Ubuntu/Debian: sudo apt install just
#   or: cargo install just
#
# Usage:
#   just              -> list all recipes
#   just build        -> configure + build
#   just test         -> run all smoke tests (same as CI)
#   just ci           -> build + test (full CI mirror)
#   just clean        -> remove build/ and stray output

set shell := ["bash", "-cu"]

# Default: list recipes
default:
    @just --list

# Configure + build in Release mode
build:
    mkdir -p build
    cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)
    @test -x build/wave_equation && echo "OK: executable built"

# Run all smoke tests (mirrors .github/workflows/build.yml)
test: build
    mkdir -p results
    @echo "Crank-Nicolson test for energy conservation:"
    mpirun --oversubscribe -np 2 build/wave_equation parameters/theta_cn.prm | tee /tmp/cn.log
    @python3 -c "import re; log=open('/tmp/cn.log').read(); er=float(re.search(r'Final E/E0:\s*([0-9.eE+-]+)', log).group(1)); assert abs(er-1.0)<1e-4, f'CN energy not conserved: E/E0={er}'; print(f'OK: CN E/E0 = {er}')"

    @echo "RK4 (energy + accuracy)"
    mpirun --oversubscribe -np 2 build/wave_equation parameters/rk4.prm | tee /tmp/rk4.log
    @python3 -c "import re; log=open('/tmp/rk4.log').read(); er=float(re.search(r'Final E/E0:\s*([0-9.eE+-]+)', log).group(1)); l2=float(re.search(r'Final L2 error:\s*([0-9.eE+-]+)', log).group(1)); assert abs(er-1.0)<1e-4, f'RK4 energy not conserved: E/E0={er}'; assert l2<1e-2, f'RK4 L2 too large: {l2}'; print(f'OK: RK4 E/E0={er}, L2={l2}')"

    @echo "Leapfrog (CFL stability at nominal dt)"
    mpirun --oversubscribe -np 2 build/wave_equation parameters/leapfrog.prm | tee /tmp/leapfrog.log
    @python3 -c "import re; log=open('/tmp/leapfrog.log').read(); er=float(re.search(r'Final E/E0:\s*([0-9.eE+-]+)', log).group(1)); assert 0.9<er<1.1, f'Leapfrog unstable: E/E0={er}'; print(f'OK: Leapfrog E/E0 = {er}')"

    @echo "MMS (manufactured solution)"
    mpirun --oversubscribe -np 2 build/wave_equation parameters/mms_cn.prm | tee /tmp/mms.log
    @python3 -c "import re; log=open('/tmp/mms.log').read(); l2=float(re.search(r'Final L2 error:\s*([0-9.eE+-]+)', log).group(1)); h1=float(re.search(r'Final H1 error:\s*([0-9.eE+-]+)', log).group(1)); assert l2<1e-2, f'MMS L2 too large: {l2}'; assert h1<1.0, f'MMS H1 too large: {h1}'; print(f'OK: MMS L2={l2}, H1={h1}')"

    @echo ""
    @echo "All tests passed."

# Check Python scripts compile and notebooks are valid JSON
lint:
    python3 -m compileall scripts/ -q && echo "OK: scripts compile"
    @for nb in notebooks/*.ipynb; do \
        python3 -c "import json; json.load(open('$nb'))" && echo "OK: $nb"; \
    done

# Full CI mirror: build + test + lint
ci: build test lint
    @echo ""
    @echo " CI mirror complete — all passed"

# Remove build artifacts and stray sweep files
clean:
    rm -rf build
    rm -f results/_*.prm results/*.vtu results/*.pvtu
    @echo "Cleaned build/ and stray results/ files"

# Quick single-scheme run for manual testing
run scheme="theta_cn":
    mpirun --oversubscribe -np 4 build/wave_equation parameters/{{scheme}}.prm
