import subprocess
import re
import matplotlib.pyplot as plt

executable = "./wave_equation"
prm_file = "../parameters/default.prm"

def update_prm(key, value):
    with open(prm_file, 'r') as f: lines = f.readlines()
    with open(prm_file, 'w') as f:
        for line in lines:
            if f"set {key} " in line: f.write(f"  set {key} = {value}\n")
            else: f.write(line)

print("Running Leapfrog Symplectic Stability Test...")
update_prm("Time stepping scheme", "ExplicitLeapfrog")
update_prm("Final time", "20.0")  # Run twice as long!
update_prm("Output frequency", "20")

result = subprocess.run(["mpirun", "--oversubscribe", "-np", "4", executable], cwd="../build", capture_output=True, text=True)

times, energies = [], []
for line in result.stdout.split('\n'):
    match = re.search(r'Time:\s*([-\d.]+),\s*L2 error:\s*[\d.e+-]+,\s*Energy:\s*([-\d.e+-]+)', line)
    if match:
        times.append(float(match.group(1)))
        energies.append(float(match.group(2)))

# Calculate Relative Energy E(t)/E(0)
e0 = energies[0]
relative_energy = [e / e0 for e in energies]

plt.figure(figsize=(10, 5))
plt.plot(times, relative_energy, color='#e74c3c', linewidth=2, label='Leapfrog Energy Drift')
plt.axhline(1.0, color='black', linestyle=':', label='Theoretical Exact Energy')
plt.title('Symplectic Stability of Explicit Leapfrog', fontsize=14, fontweight='bold')
plt.xlabel('Simulation Time (s)')
plt.ylabel('Relative Energy $E_t / E_0$')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('../results/leapfrog/symplectic_stability.png', dpi=300)
print("Saved to results/leapfrog/symplectic_stability.png")