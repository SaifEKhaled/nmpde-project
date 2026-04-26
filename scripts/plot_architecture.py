import json, matplotlib.pyplot as plt, numpy as np, math

print("Rendering Architecture Visualizations...")
with open("../results/sparse/architecture_cache.json", "r") as f: data = json.load(f)

# Viz 1: The Cache Sweet Spot
plt.figure(figsize=(6,6))
bars = plt.bar(['Traditional Sparse\n(Cache Resident)', 'Matrix-Free SIMD\n(Compute Bound)'], 
               [data["exp1"]["sparse"], data["exp1"]["mf"]], color=['#7f8c8d', '#27ae60'], width=0.5)
for bar in bars:
    plt.annotate(f'{bar.get_height():.2f} MDoPS', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 5), textcoords="offset points", ha='center', fontweight='bold')
plt.ylabel('Throughput (Million DoFs Per Second)'); plt.title("Viz 1: The L3 Cache Baseline", fontweight='bold')
plt.ylim(0, max(data["exp1"]["sparse"], data["exp1"]["mf"]) * 1.3); plt.grid(axis='y', ls='--', alpha=0.5)
plt.tight_layout(); plt.savefig("../results/sparse/viz_1_cache.png", dpi=300)

# Viz 2: The Memory Wall
plt.figure(figsize=(8,5))
dofs = data["exp2"]["dofs"]
plt.plot(dofs, data["exp2"]["sparse"], '-o', color='#c0392b', linewidth=2.5, markersize=8, label='Sparse Matrix (RAM Bottleneck)')
plt.plot(dofs, data["exp2"]["mf"], '-s', color='#27ae60', linewidth=2.5, markersize=8, label='Matrix-Free (SIMD Vectorized)')
plt.xscale('log'); plt.xlabel("Degrees of Freedom (Grid Resolution)"); plt.ylabel("Throughput (MDoPS)")
plt.title("Viz 2: The Von Neumann Memory Wall", fontweight='bold')
plt.axvline(x=25000, color='black', linestyle=':', label='L3 Cache Capacity Limit')
plt.grid(True, which="both", ls="--", alpha=0.4); plt.legend()
plt.tight_layout(); plt.savefig("../results/sparse/viz_2_throughput.png", dpi=300)

# Viz 3: Memory Footprint Scaling (Theoretical bytes)
plt.figure(figsize=(6,6))
ram_sparse = [d * math.log(d)* 80 / (1024**2) for d in dofs] # ~O(N log N) for 2D sparse
ram_mf = [d * 8 / (1024**2) for d in dofs] # O(N) purely vectors
plt.loglog(dofs, ram_sparse, '-o', color='#8e44ad', linewidth=2, label='Sparse Matrix Memory Allocation')
plt.loglog(dofs, ram_mf, '-s', color='#f39c12', linewidth=2, label='Matrix-Free Vector Storage')
plt.xlabel("Degrees of Freedom"); plt.ylabel("Memory Footprint (Megabytes)")
plt.title("Viz 3: Asymptotic Hardware Footprint", fontweight='bold')
plt.grid(True, which="both", ls="--", alpha=0.4); plt.legend()
plt.tight_layout(); plt.savefig("../results/sparse/viz_3_ram.png", dpi=300)

# Viz 4: MPI Strong Scaling Efficiency
plt.figure(figsize=(6,6))
ranks = data["exp4"]["ranks"]
plt.plot(ranks, data["exp4"]["efficiency"], '-o', color='#2980b9', linewidth=2.5, markersize=8)
plt.axhline(100, color='black', linestyle='--', label='Ideal Linear Scaling (100%)')
plt.xlabel("MPI Execution Ranks (Cores)"); plt.ylabel("Parallel Efficiency (%)")
plt.title("Viz 4: Amdahl's Law & Communication Plateau", fontweight='bold')
plt.xticks(ranks); plt.grid(True, ls="--", alpha=0.5); plt.legend()
plt.tight_layout(); plt.savefig("../results/sparse/viz_4_scaling.png", dpi=300)

print("All 4 Architecture visuals generated instantly!")