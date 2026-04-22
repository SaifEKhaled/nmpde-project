import subprocess
import re
import matplotlib.pyplot as plt

# --- Configuration ---
executable = "./wave_equation"
prm_file = "../parameters/default.prm"
schemes = ["ExplicitLeapfrog", "RungeKutta4"]

# Dictionary to hold our extracted data
data = {scheme: {"time": [], "energy": []} for scheme in schemes}

print("Starting Automated Physics/Energy Sweep...")

# --- Helper function to modify the .prm file dynamically ---
def set_scheme(scheme_name):
    with open(prm_file, 'r') as file:
        lines = file.readlines()
    
    with open(prm_file, 'w') as file:
        for line in lines:
            if "set Time stepping scheme" in line:
                file.write(f"  set Time stepping scheme = {scheme_name}\n")
            else:
                file.write(line)

# --- Run the Sweep ---
for scheme in schemes:
    print(f"Running Simulation with {scheme}... ", end="", flush=True)
    
    # Update the parameter file
    set_scheme(scheme)
    
    # Run the C++ engine on 4 cores (fast enough, but stable)
    result = subprocess.run(
        ["mpirun", "-np", "4", executable],
        cwd="../build",
        capture_output=True,
        text=True
    )
    
    # Extract Time and Energy using Regex
    # Matches lines like: "   Time:    0.5, L2 error: 1.2e-08, Energy: 2.451"
    matches = re.finditer(r"Time:\s*([-0-9.]+),\s*L2 error:.*Energy:\s*([0-9.]+)", result.stdout)
    
    count = 0
    for match in matches:
        data[scheme]["time"].append(float(match.group(1)))
        data[scheme]["energy"].append(float(match.group(2)))
        count += 1
        
    print(f"Captured {count} data points.")

# --- Plotting the Results ---
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-darkgrid')

colors = {'ExplicitLeapfrog': 'red', 'RungeKutta4': 'blue', 'ThetaMethod': 'green'}
labels = {'ExplicitLeapfrog': 'Leapfrog (2nd Order)', 'RungeKutta4': 'RK4 (4th Order)', 'ThetaMethod': 'Crank-Nicolson (Implicit)'}

for scheme in schemes:
    # Only plot if we successfully captured data
    if data[scheme]["time"]:
        # Normalize energy so they all start at exactly 1.0 (relative energy drift)
        initial_energy = data[scheme]["energy"][0]
        normalized_energy = [e / initial_energy for e in data[scheme]["energy"]]
        
        plt.plot(data[scheme]["time"], normalized_energy, label=labels[scheme], color=colors[scheme], linewidth=2)

# Aesthetics
plt.title("Hamiltonian Energy Conservation Over Time", fontsize=14, fontweight='bold')
plt.xlabel("Simulation Time (s)", fontsize=12)
plt.ylabel("Relative Energy ($E_t / E_0$)", fontsize=12)
# Zoom in tight on the Y-axis to show the microscopic drift
plt.ylim(0.99, 1.01) 
plt.legend(fontsize=11)

# Save the plot
plt.savefig("../results/energy_conservation.png", dpi=300, bbox_inches='tight')
print("\nSweep complete! Graph saved to results/energy_conservation.png")