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
        
        # Build the command - Note: added --oversubscribe for WSL stability
        cmd = ["mpirun", "--oversubscribe", "-np", str(cores), executable]
        
        # Run the C++ code and capture the output
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # NEW REGEX: Looking for "Throughput: XX.XX MDoPS"
        match = re.search(r'Throughput:\s+([\d.]+)\s+MDoPS', result.stdout)
        
        if match:
            throughput = float(match.group(1))
            # Since your CSV wants "ComputeTime", we use the inverse of throughput.
            # (Higher throughput = Lower compute time)
            # We'll treat 1/throughput as the 'normalized' compute time for your graph.
            compute_time = 1.0 / throughput 
            
            writer.writerow([cores, compute_time])
            print(f"Done! ({throughput} MDoPS -> {compute_time:.4f} relative sec)")
        else:
            print("Failed! Regex mismatch. Raw output snippet:")
            # This helps you see what the C++ code is actually saying
            print(result.stdout.strip().split('\n')[-1]) 
            writer.writerow([cores, "ERROR"])

print(f"Benchmark complete! Data saved to {csv_filename}")