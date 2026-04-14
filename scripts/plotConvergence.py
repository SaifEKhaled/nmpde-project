import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_convergence_plot():
    file_path = "../data/convergence_results.csv"
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}. Run convergenceStudy.py first.")
        return

    # Normalize column names dynamically in case they vary slightly
    df.columns = df.columns.str.strip().str.lower()
    error_col = [c for c in df.columns if 'error' in c][0]
    ref_col = [c for c in df.columns if 'refinement' in c][0]

    # Calculate mesh size h. (h is proportional to 1 / 2^Refinement)
    df['h'] = 1.0 / (2 ** df[ref_col])

    plt.figure(figsize=(9, 6))

    # Plot the actual simulation data
    plt.loglog(df['h'], df[error_col], 'bo-', linewidth=2.5, markersize=8, label='Matrix-Free L2 Error')

    # Create a theoretical O(h^5) reference line anchored to the first data point
    h_ref = df['h']
    error_ref = df[error_col].iloc[0] * (h_ref / h_ref.iloc[0])**5

    plt.loglog(h_ref, error_ref, 'k--', linewidth=2, alpha=0.7, label='Theoretical O(h^5) Reference')

    plt.title("Spatial Convergence Study (Q4 Spectral Elements)", fontsize=15, fontweight='bold')
    plt.xlabel("Mesh Size (h)", fontsize=12)
    plt.ylabel("L2 Error Norm", fontsize=12)
    plt.legend(fontsize=12)
    
    # Grid lines are crucial for log-log plots
    plt.grid(True, which="both", ls="--", alpha=0.5)

    os.makedirs("../results", exist_ok=True)
    plt.savefig("../results/convergence_plot.png", dpi=300, bbox_inches='tight')
    print("\nSUCCESS: Convergence plot saved to results/convergence_plot.png")

if __name__ == "__main__":
    generate_convergence_plot()