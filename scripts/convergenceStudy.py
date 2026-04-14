import subprocess
import re
import pandas as pd

refinements = [4, 5, 6]
results = []

for r in refinements:
    print(f"Running Refinement Level {r}...")
    # Update the .prm file on the fly
    with open("../parameters/default.prm", "r") as f:
        lines = f.readlines()
    
    with open("../parameters/default.prm", "w") as f:
        for line in lines:
            if "Global refinement" in line:
                f.write(f"set Global refinement = {r}\n")
            else:
                f.write(line)
    
    # Run the code
    proc = subprocess.run(["mpirun", "-np", "4", "./wave_equation"], 
                          capture_output=True, text=True)
    
    # Find the final error
    # Look for: "Time: 10, solution L2 error: XXX"
    errors = re.findall(r"L2 error:\s+([\d.e+-]+)", proc.stdout)
    if errors:
        final_error = float(errors[-1])
        results.append({"Refinement": r, "Error": final_error})
        print(f"   Error: {final_error}")

df = pd.DataFrame(results)
df.to_csv("../data/convergence_results.csv", index=False)
print("Saved to data/convergence_results.csv")