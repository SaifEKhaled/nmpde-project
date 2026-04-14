import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def run_energy_test(scheme):
    print(f"Testing Energy Conservation for {scheme}...")
    prm_path = "../parameters/default.prm"
    with open(prm_path, "r") as f:
        lines = f.readlines()
    with open(prm_path, "w") as f:
        for line in lines:
            if "Time stepping scheme" in line:
                f.write(f"set Time stepping scheme = {scheme}\n")
            else:
                f.write(line)
    
    # Using -np 1 for consistent parsing in the script
    proc = subprocess.run(["mpirun", "-np", "1", "./wave_equation"], capture_output=True, text=True)
    
    # This Regex matches your exact C++ output: "Time: -10, L2 error: 5.47e-09, Energy: 2.466"
    # It accounts for any amount of whitespace and scientific notation
    pattern = r"Time:\s*([\d.e+-]+),\s*L2 error:\s*[\d.e+-]+,\s*Energy:\s*([\d.e+-]+)"
    matches = re.findall(pattern, proc.stdout)
    
    if not matches:
        print(f"   Warning: No data matched for {scheme}. Check C++ output format!")
        return pd.DataFrame(columns=["Time", "Energy"])

    data = [{"Time": float(t), "Energy": float(e)} for t, e in matches]
    print(f"   Parsed {len(data)} points.")
    return pd.DataFrame(data)

df_lf = run_energy_test("ExplicitLeapfrog")
df_rk = run_energy_test("RungeKutta4")

if not df_lf.empty and not df_rk.empty:
    plt.figure(figsize=(10, 6))
    initial_e = df_rk['Energy'].iloc[0]
    
    # Relative Error
    rel_err_lf = (df_lf['Energy'] - initial_e) / initial_e
    rel_err_rk = (df_rk['Energy'] - initial_e) / initial_e
    
    # SAFETY: If RK4 has exploded (values > 1e-1), we only plot Leapfrog
    # to keep the graph readable.
    if rel_err_rk.abs().max() > 1e-1:
        print("   Notice: RK4 unstable at this CFL. Plotting Leapfrog only.")
        plt.plot(df_lf['Time'], rel_err_lf, 'r--', label='Leapfrog (Stable)')
        plt.annotate('RK4 Exploded (>1e50)', xy=(df_rk['Time'].iloc[-1], 0), 
                     color='blue', fontweight='bold')
    else:
        plt.plot(df_lf['Time'], rel_err_lf, 'r--', label='Leapfrog (2nd Order)')
        plt.plot(df_rk['Time'], rel_err_rk, 'b-', alpha=0.8, label='RK4 (4th Order)')
    
    plt.title("Relative Energy Drift: Stability Analysis", fontsize=14, fontweight='bold')
    plt.xlabel("Physical Time ($t$)", fontsize=12)
    plt.ylabel("Relative Error $(E(t) - E_0) / E_0$", fontsize=12)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.savefig("../results/stability_analysis.png", dpi=300)
    print("\nDONE: Check results/stability_analysis.png")
else:
    print("\nERROR: Dataframes empty. Ensure C++ is printing 'Time: X, L2 error: Y, Energy: Z'")