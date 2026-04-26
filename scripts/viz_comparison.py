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

print("Running Architecture Throughput Duel...")
update_prm("Time stepping scheme", "RungeKutta4")
update_prm("Final time", "1.0")
update_prm("Global refinement", "5")

methods = ["SparseMatrix", "MatrixFree"]
mdops = []

for method in methods:
    update_prm("Assembly Method", method)
    print(f"  Benchmarking {method}...", end="", flush=True)
    result = subprocess.run(["mpirun", "--oversubscribe", "-np", "4", executable], cwd="../build", capture_output=True, text=True)
    
    match = re.search(r'Throughput:\s+([\d.]+)\s+MDoPS', result.stdout)
    if match:
        mdops.append(float(match.group(1)))
        print(f" {mdops[-1]} MDoPS")

# Plotting
plt.figure(figsize=(8, 6))
bars = plt.bar(['Traditional Sparse\n(Memory Bound)', 'Matrix-Free SIMD\n(Compute Bound)'], mdops, color=['#7f8c8d', '#27ae60'], width=0.5)

for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.2f} MDoPS', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylabel('Throughput (Million DoFs Per Second)', fontsize=12)
plt.title('Algorithmic Architecture Comparison (4 MPI Ranks)', fontsize=14, fontweight='bold')
plt.ylim(0, max(mdops) * 1.3)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('../results/sparse/throughput_comparison.png', dpi=300)
print("Saved to results/sparse/throughput_comparison.png")