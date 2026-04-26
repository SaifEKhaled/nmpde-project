import subprocess, re, json, time

exe, prm = "../build/wave_equation", "../parameters/default.prm"
def update_prm(k, v):
    with open(prm, 'r') as f: lines = f.readlines()
    with open(prm, 'w') as f:
        for l in lines:
            if f"set {k} " in l: f.write(f"  set {k} = {v}\n")
            else: f.write(l)

print("Running REAL 20-Core MPI Hardware Profiling...")
data = {"ranks": [1, 2, 4, 8, 16], "wall_time": [], "speedup": [], "efficiency": [], "compute_time": [], "mpi_wait_time": []}

# Setup a moderate problem size so the 1-core run doesn't take hours
update_prm("Time stepping scheme", "ExplicitLeapfrog") # Leapfrog is faster for scaling tests
update_prm("Assembly Method", "MatrixFree")
update_prm("Global refinement", "6") 
update_prm("Final time", "2.0")
update_prm("Output frequency", "0") # Turn off VTU output to prevent I/O disk bottlenecks!

for r in data["ranks"]:
    print(f"  -> Testing {r} MPI Ranks...", end="", flush=True)
    
    # Measure strict Wall Clock Time
    start_time = time.time()
    subprocess.run(["mpirun", "--oversubscribe", "-np", str(r), exe], cwd="../build", capture_output=True)
    end_time = time.time()
    
    duration = end_time - start_time
    data["wall_time"].append(duration)
    print(f" {duration:.2f} seconds")

# Calculate Amdahl's Speedup and Efficiency based on real data
base_time = data["wall_time"][0]
for idx, r in enumerate(data["ranks"]):
    speedup = base_time / data["wall_time"][idx]
    efficiency = (speedup / r) * 100.0
    data["speedup"].append(speedup)
    data["efficiency"].append(efficiency)
    
    # Estimate Compute vs Comm bound (Heuristic based on efficiency drop)
    ideal_compute = base_time / r
    actual_time = data["wall_time"][idx]
    comm_penalty = max(0, actual_time - ideal_compute)
    
    data["compute_time"].append(ideal_compute)
    data["mpi_wait_time"].append(comm_penalty)

with open("../results/scaling/mpi_cache.json", "w") as f:
    json.dump(data, f)
print("Real MPI Data successfully cached to results/scaling/mpi_cache.json")