#!/usr/bin/env bash
set -e

echo "=== Building ==="

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
cd ..

echo "=== CN ==="
mpirun --oversubscribe -np 2 \
  build/wave_equation parameters/theta_cn.prm > cn.log

echo "=== RK4 ==="
mpirun --oversubscribe -np 2 \
  build/wave_equation parameters/rk4.prm > rk4.log

echo "=== Leapfrog ==="
mpirun --oversubscribe -np 2 \
  build/wave_equation parameters/leapfrog.prm > leapfrog.log

echo "=== MMS ==="
mpirun --oversubscribe -np 2 \
  build/wave_equation parameters/mms_cn.prm > mms.log

echo "=== Python checks ==="
python3 -m compileall scripts/ -q

echo "=== Validate logs ==="
python3 <<'EOF'
import re

def check_cn():
    log = open("cn.log").read()
    er = float(re.search(r"Final E/E0:\s*([\d.eE+\-]+)", log).group(1))
    assert abs(er - 1.0) < 1e-4

def check_rk4():
    log = open("rk4.log").read()
    er = float(re.search(r"Final E/E0:\s*([\d.eE+\-]+)", log).group(1))
    l2 = float(re.search(r"Final L2 error:\s*([\d.eE+\-]+)", log).group(1))
    assert abs(er - 1.0) < 1e-4
    assert l2 < 1e-2

def check_leapfrog():
    log = open("leapfrog.log").read()
    er = float(re.search(r"Final E/E0:\s*([\d.eE+\-]+)", log).group(1))
    assert 0.9 < er < 1.1

def check_mms():
    log = open("mms.log").read()
    l2 = float(re.search(r"Final L2 error:\s*([\d.eE+\-]+)", log).group(1))
    h1 = float(re.search(r"Final H1 error:\s*([\d.eE+\-]+)", log).group(1))
    assert l2 < 1e-2
    assert h1 < 1.0

check_cn()
check_rk4()
check_leapfrog()
check_mms()

print("All tests passed.")
EOF

echo "=== SUCCESS ==="