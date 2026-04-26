import subprocess, re, json

exe, prm = "../build/wave_equation", "../parameters/default.prm"
def update_prm(k, v):
    with open(prm, 'r') as f: lines = f.readlines()
    with open(prm, 'w') as f:
        for l in lines:
            if f"set {k} " in l: f.write(f"  set {k} = {v}\n")
            else: f.write(l)

print("Generating RK4 Data Cache (This will take a minute)...")
data = {"exp1": {"t": [], "e": []}, "exp3": {"cfl": [], "err": []}, "exp4": {"cfl": [], "err": []}}

# 1. LONG-TERM DRIFT (Proving it is NOT symplectic)
update_prm("Time stepping scheme", "RungeKutta4")
update_prm("Final time", "10.0")
update_prm("Output frequency", "2")
update_prm("CFL number", "0.5")
res = subprocess.run(["mpirun", "--oversubscribe", "-np", "4", exe], cwd="../build", capture_output=True, text=True)

for line in res.stdout.split('\n'):
    m = re.search(r'Time:\s*([-\d.]+),.*Energy:\s*([-\d.e+-]+)', line)
    if m: 
        data["exp1"]["t"].append(float(m.group(1)))
        data["exp1"]["e"].append(float(m.group(2)))

# 3. THE SPATIAL ERROR FLOOR (Tiny time steps)
update_prm("Final time", "2.0")
cfl_floor = [0.05, 0.1, 0.2, 0.4]
for cfl in cfl_floor:
    update_prm("CFL number", str(cfl))
    res = subprocess.run(["mpirun", "--oversubscribe", "-np", "4", exe], cwd="../build", capture_output=True, text=True)
    m = re.findall(r'L2 error:\s*([\d.e+-]+)', res.stdout)
    if m:
        data["exp3"]["cfl"].append(cfl)
        data["exp3"]["err"].append(float(m[-1]))

# 4. THE TEMPORAL CONVERGENCE (Large time steps to break the floor)
cfl_true = [1.8, 2.2, 2.6, 2.8]
for cfl in cfl_true:
    update_prm("CFL number", str(cfl))
    res = subprocess.run(["mpirun", "--oversubscribe", "-np", "4", exe], cwd="../build", capture_output=True, text=True)
    m = re.findall(r'L2 error:\s*([\d.e+-]+)', res.stdout)
    if m:
        data["exp4"]["cfl"].append(cfl)
        data["exp4"]["err"].append(float(m[-1]))

with open("../results/rk4/rk4_cache.json", "w") as f:
    json.dump(data, f)
print("Data successfully cached to results/rk4/rk4_cache.json")