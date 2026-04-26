import json, matplotlib.pyplot as plt, numpy as np

print("Rendering Leapfrog Visualizations...")
with open("../results/leapfrog/leapfrog_cache.json", "r") as f:
    data = json.load(f)

# Viz 1: Energy Conservation
plt.figure(figsize=(8,4))
e_rel = [e / data["exp1"]["e"][0] for e in data["exp1"]["e"]]
plt.plot(data["exp1"]["t"], e_rel, color='#e74c3c', linewidth=1.5, alpha=0.9)
plt.axhline(1.0, color='black', linestyle='--', linewidth=1)
plt.title("Viz 1: Long-Term Hamiltonian Conservation", fontweight='bold')
plt.xlabel("Time (s)"); plt.ylabel("Relative Energy ($E_t / E_0$)")
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("../results/leapfrog/viz_1_energy.png", dpi=300)

# Viz 2: Phase Space Portrait (Takens' Delay Embedding)
plt.figure(figsize=(6,6))
# We plot Energy at time t vs Energy at time t+5 to create a topology
delay = 5
u = e_rel[:-delay]
v = e_rel[delay:]
plt.plot(u, v, color='#8e44ad', linewidth=1.0, alpha=0.7)
plt.title("Viz 2: Delay-Embedded Phase Topology", fontweight='bold')
plt.xlabel("Energy state $E(t)$")
plt.ylabel(f"Energy state $E(t + {delay}\\Delta t)$")
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("../results/leapfrog/viz_2_phase.png", dpi=300)

# Viz 3: Convergence
plt.figure(figsize=(6,6))
cfls, errs = data["exp3"]["cfl"], data["exp3"]["err"]
slope = np.polyfit(np.log(cfls), np.log(errs), 1)[0]
plt.loglog(cfls, errs, '-o', color='#2980b9', linewidth=2, markersize=8, label=f'Empirical Slope: {slope:.2f}')
plt.title("Viz 3: 2nd-Order Temporal Convergence", fontweight='bold')
plt.xlabel("CFL Number ($\propto \Delta t$)"); plt.ylabel("$L_2$ Error Norm")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()
plt.tight_layout(); plt.savefig("../results/leapfrog/viz_3_convergence.png", dpi=300)

# Viz 4: The CFL Explosion
plt.figure(figsize=(8,4))
valid_t, valid_e = [], []
for t, e in zip(data["exp4"]["t"], data["exp4"]["e"]):
    if e > 0 and not np.isnan(e): 
        valid_t.append(t)
        valid_e.append(e)

plt.semilogy(valid_t, valid_e, color='#c0392b', linewidth=2)
plt.title("Viz 4: The CFL Limit (Numerical Explosion)", fontweight='bold')
plt.xlabel("Time (s)"); plt.ylabel("Total Energy (Log Scale)")
plt.grid(True, alpha=0.3)
# Safely add text without breaking layout bounds
plt.text(valid_t[len(valid_t)//2], max(valid_e)*0.1, 'Energy diverges to infinity\n(Instability Triggered)', 
         ha='center', va='center', color='black', fontsize=11, fontweight='bold', 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
plt.tight_layout(); plt.savefig("../results/leapfrog/viz_4_cfl.png", dpi=300)

print("All 4 high-quality visuals generated instantly!")