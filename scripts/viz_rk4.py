import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np

executable = "./wave_equation"
prm_file = "../parameters/default.prm"

def update_prm(key, value):
    with open(prm_file, 'r') as f: lines = f.readlines()
    with open(prm_file, 'w') as f:
        for line in lines:
            if f"set {key} " in line: f.write(f"  set {key} = {value}\n")
            else: f.write(line)

print("Running RK4 Convergence Analysis...")
update_prm("Time stepping scheme", "RungeKutta4")
update_prm("Final time", "2.0") # Shorter time, focus on accuracy

cfl_values = [1.2, 1.0, 0.8, 0.6]

errors = []

for cfl in cfl_values:
    update_prm("CFL number", str(cfl))
    print(f"  Testing CFL = {cfl}...", end="", flush=True)
    result = subprocess.run(["mpirun", "--oversubscribe", "-np", "4", executable], cwd="../build", capture_output=True, text=True)
    
    # Grab the LAST L2 error reported
    matches = re.findall(r'L2 error:\s*([\d.e+-]+)', result.stdout)
    if matches:
        errors.append(float(matches[-1]))
        print(f" Error: {errors[-1]:.2e}")

# Calculate empirical slope (Order of convergence)
log_dt = np.log(cfl_values)
log_err = np.log(errors)
slope, _ = np.polyfit(log_dt, log_err, 1)

plt.figure(figsize=(8, 6))
plt.loglog(cfl_values, errors, marker='o', markersize=8, color='#2980b9', linewidth=2, label=f'RK4 Error Data (Slope: {slope:.2f})')
plt.title('Runge-Kutta 4: Temporal Convergence Order', fontsize=14, fontweight='bold')
plt.xlabel('Time Step factor ($\propto \Delta t$)')
plt.ylabel('$L_2$ Norm of Error')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('../results/rk4/convergence_order.png', dpi=300)
print(f"Saved to results/rk4/convergence_order.png. Empirical Order: {slope:.2f}")