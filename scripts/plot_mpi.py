import json, matplotlib.pyplot as plt, numpy as np

print("Rendering MPI Cluster Visualizations...")
with open("../results/scaling/mpi_cache.json", "r") as f: data = json.load(f)
ranks = data["ranks"]

# Viz 1: Strong Scaling (Wall Time)
plt.figure(figsize=(6,6))
plt.loglog(ranks, data["wall_time"], '-o', color='#8e44ad', linewidth=2.5, markersize=8, label='Actual Wall Time')
plt.loglog(ranks, [data["wall_time"][0]/r for r in ranks], 'k--', alpha=0.6, label='Ideal Linear Time')
plt.xticks(ranks, ranks); plt.xlabel("MPI Ranks (Cores)"); plt.ylabel("Wall Clock Time (Seconds)")
plt.title("Viz 1: MPI Strong Scaling (Wall Time)", fontweight='bold')
plt.grid(True, which="both", ls="--", alpha=0.4); plt.legend()
plt.tight_layout(); plt.savefig("../results/scaling/viz_1_time.png", dpi=300)

# Viz 2: The Speedup Curve (Amdahl's Law)
plt.figure(figsize=(6,6))
plt.plot(ranks, data["speedup"], '-o', color='#2980b9', linewidth=2.5, markersize=8, label='Empirical Speedup')
plt.plot(ranks, ranks, 'k--', alpha=0.6, label='Ideal Speedup ($S = N$)')
plt.xlabel("MPI Ranks"); plt.ylabel("Speedup Factor"); plt.title("Viz 2: Amdahl's Law & Cluster Speedup", fontweight='bold')
plt.grid(True, ls="--", alpha=0.4); plt.legend()
plt.tight_layout(); plt.savefig("../results/scaling/viz_2_speedup.png", dpi=300)

# Viz 3: Parallel Efficiency (The Super-Linear Spike)
plt.figure(figsize=(6,6))
plt.plot(ranks, data["efficiency"], '-o', color='#27ae60', linewidth=2.5, markersize=8)
plt.axhline(100, color='black', linestyle='--')
plt.xlabel("MPI Ranks"); plt.ylabel("Parallel Efficiency (%)"); plt.title("Viz 3: Super-Linear Cache Efficiency", fontweight='bold')
plt.annotate('L3 Cache\nAlignment!', xy=(4, 190), xytext=(4, 150), ha='center',
             arrowprops=dict(facecolor='black', shrink=0.05), fontweight='bold')
plt.grid(True, ls="--", alpha=0.4)
plt.tight_layout(); plt.savefig("../results/scaling/viz_3_efficiency.png", dpi=300)

# Viz 4: Hardware Profiling (Compute vs Comm)
plt.figure(figsize=(8,5))
plt.bar(ranks, data["compute_time"], color='#2c3e50', label='SIMD Compute', width=0.8)
plt.bar(ranks, data["mpi_wait_time"], bottom=data["compute_time"], color='#e74c3c', label='MPI Comm / Latency', width=0.8)
plt.xticks(ranks); plt.xlabel("MPI Ranks"); plt.ylabel("Time (Seconds)"); plt.title("Viz 4: The Network Bottleneck", fontweight='bold')
plt.legend(); plt.grid(axis='y', ls="--", alpha=0.3)
plt.tight_layout(); plt.savefig("../results/scaling/viz_4_profile.png", dpi=300)

print("All 4 MPI Cluster visuals generated instantly!")