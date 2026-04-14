import subprocess
import re
import csv
import os

# Ensure we are in the build directory or path to executable is correct
executable = "./wave_equation"
if not os.path.exists(executable):
    print(f"Error: {executable} not found. Make sure you are in the build folder.")
    exit(1)

cores_to_test = [1, 2, 4, 8, 16]
csv_filename = "scaling_data.csv"

print("Starting Automated HPC Scaling Benchmark...")

# Open CSV file to write results
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Cores", "ComputeTime_sec"]) # Header

    for cores in cores_to_test:
        print(f"Running on {cores} core(s)... ", end="", flush=True)
        
        # Build the command
        cmd = ["mpirun", "--use-hwthread-cpus", "-np", str(cores), executable]
        
        # Run the C++ code and capture the output
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Use Regex to find the exact computation time line
        # Looking for: "Spent XX.XXs on output and YY.YYs on computations."
        match = re.search(r'and ([\d.]+)s on computations', result.stdout)
        
        if match:
            compute_time = float(match.group(1))
            writer.writerow([cores, compute_time])
            print(f"Done! ({compute_time} seconds)")
        else:
            print("Failed! Could not find computation time in output.")
            writer.writerow([cores, "ERROR"])

print(f"✅ Benchmark complete! Data saved to {csv_filename}")