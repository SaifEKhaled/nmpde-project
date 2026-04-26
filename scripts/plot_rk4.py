import json, matplotlib.pyplot as plt, numpy as np

print("Rendering RK4 Visualizations...")
with open("../results/rk4/rk4_cache.json", "r") as f:
    data = json.load(f)

# Viz 1: Energy Drift (Notice the Y-axis won't be perfectly flat like Leapfrog)
plt.figure(figsize=(8,4))
e_rel = [e / data["exp1"]["e"][0] for e in data["exp1"]["e"]]
plt.plot(data["exp1"]["t"], e_rel, color='#2980b9', linewidth=2)
plt.axhline(1.0, color='black', linestyle='--', linewidth=1)
plt.title("Viz 1: Non-Symplectic Energy Drift", fontweight='bold')
plt.xlabel("Time (s)"); plt.ylabel("Relative Energy ($E_t / E_0$)")
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("../results/rk4/viz_1_energy.png", dpi=300)

# Viz 2: Phase Space (Inward Spiral)
plt.figure(figsize=(6,6))
delay = 5
u, v = e_rel[:-delay], e_rel[delay:]
plt.plot(u, v, color='#2c3e50', linewidth=1.0, alpha=0.8)
plt.title("Viz 2: Dissipative Phase Topology", fontweight='bold')
plt.xlabel("Energy state $E(t)$"); plt.ylabel(f"Energy state $E(t + {delay}\\Delta t)$")
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("../results/rk4/viz_2_phase.png", dpi=300)

# Viz 3: The Spatial Error Floor
plt.figure(figsize=(6,6))
cfls, errs = data["exp3"]["cfl"], data["exp3"]["err"]
slope = np.polyfit(np.log(cfls), np.log(errs), 1)[0]
plt.loglog(cfls, errs, '-o', color='#7f8c8d', linewidth=2, markersize=8, label=f'Apparent Slope: {slope:.2f}')
plt.title("Viz 3: The Spatial Error Floor", fontweight='bold')
plt.xlabel("CFL Number ($\propto \Delta t$)"); plt.ylabel("$L_2$ Error Norm")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()
plt.tight_layout(); plt.savefig("../results/rk4/viz_3_floor.png", dpi=300)

# Viz 4: True Convergence
plt.figure(figsize=(6,6))
cfls, errs = data["exp4"]["cfl"], data["exp4"]["err"]
slope = np.polyfit(np.log(cfls), np.log(errs), 1)[0]
plt.loglog(cfls, errs, '-o', color='#27ae60', linewidth=2, markersize=8, label=f'True Slope: {slope:.2f}')
plt.title("Viz 4: True Temporal Convergence", fontweight='bold')
plt.xlabel("CFL Number ($\propto \Delta t$)"); plt.ylabel("$L_2$ Error Norm")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()
plt.tight_layout(); plt.savefig("../results/rk4/viz_4_convergence.png", dpi=300)

print("All 4 RK4 visuals generated instantly!")