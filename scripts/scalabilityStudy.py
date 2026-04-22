import subprocess
import re
import matplotlib.pyplot as plt

# --- The "Just What You Need" Configuration ---
executable = "./wave_equation"
cores_to_test = [1, 2, 4, 8, 16]  # The standard academic powers of 2
throughput_results = []

print("Starting Standard HPC Strong Scaling Sweep...")

for p in cores_to_test:
    print(f"Running on {p:2} MPI rank(s)... ", end="", flush=True)
    
    result = subprocess.run(
        ["mpirun", "--oversubscribe", "-np", str(p), executable],
        cwd="../build",
        capture_output=True,
        text=True
    )
    
    match = re.search(r"Throughput:\s+([0-9.]+)\s+MDoPS", result.stdout)
    
    if match:
        mdops = float(match.group(1))
        throughput_results.append(mdops)
        print(f"Achieved {mdops:>6.2f} MDoPS")
    else:
        print("ERROR: Could not parse throughput!")
        throughput_results.append(0)

# --- Calculate Speedup ---
base_throughput = throughput_results[0]
actual_speedup = [throughput / base_throughput for throughput in throughput_results]
ideal_speedup = cores_to_test

# --- Plotting the Results ---
plt.figure(figsize=(8, 5))
plt.style.use('seaborn-v0_8-whitegrid')
plt.plot(cores_to_test, ideal_speedup, 'k--', label="Ideal Linear Scaling")
plt.plot(cores_to_test, actual_speedup, 'bo-', label="Matrix-Free SIMD Speedup")
plt.title("Strong Scaling Analysis", fontweight='bold')
plt.xlabel("MPI Ranks")
plt.ylabel("Speedup")
plt.xticks(cores_to_test)
plt.legend()
plt.savefig("../results/amdahls_law_scaling.png", dpi=300, bbox_inches='tight')
print("\nSweep complete! Graph saved.")