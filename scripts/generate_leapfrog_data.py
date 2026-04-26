import subprocess, re, json

exe, prm = "../build/wave_equation", "../parameters/default.prm"
def update_prm(k, v):
    with open(prm, 'r') as f: lines = f.readlines()
    with open(prm, 'w') as f:
        for l in lines:
            if f"set {k} " in l: f.write(f"  set {k} = {v}\n")
            else: f.write(l)

print("Generating Leapfrog Data Cache...")
data = {"exp1": {"t": [], "e": []}, "exp3": {"cfl": [], "err": []}, "exp4": {"t": [], "e": []}}

# 1. LONG-TERM STABILITY (Using energy for phase space too)
update_prm("Time stepping scheme", "ExplicitLeapfrog")
update_prm("Final time", "10.0")
update_prm("Output frequency", "2")
update_prm("CFL number", "0.5")
res = subprocess.run(["mpirun", "--oversubscribe", "-np", "4", exe], cwd="../build", capture_output=True, text=True)

for line in res.stdout.split('\n'):
    m = re.search(r'Time:\s*([-\d.]+),.*Energy:\s*([-\d.e+-]+)', line)
    if m: 
        data["exp1"]["t"].append(float(m.group(1)))
        data["exp1"]["e"].append(float(m.group(2)))

# 3. TEMPORAL CONVERGENCE
cfl_list = [0.8, 0.4, 0.2, 0.1]
for cfl in cfl_list:
    update_prm("CFL number", str(cfl))
    res = subprocess.run(["mpirun", "--oversubscribe", "-np", "4", exe], cwd="../build", capture_output=True, text=True)
    m = re.findall(r'L2 error:\s*([\d.e+-]+)', res.stdout)
    if m:
        data["exp3"]["cfl"].append(cfl)
        data["exp3"]["err"].append(float(m[-1]))

# 4. CFL INSTABILITY (The True Explosion)
update_prm("CFL number", "1.5") # Pushed to guaranteed failure!
update_prm("Final time", "2.0") # It will fail quickly
res = subprocess.run(["mpirun", "--oversubscribe", "-np", "4", exe], cwd="../build", capture_output=True, text=True)
for line in res.stdout.split('\n'):
    m = re.search(r'Time:\s*([-\d.]+),.*Energy:\s*([-\d.e+-]+)', line)
    if m:
        data["exp4"]["t"].append(float(m.group(1)))
        data["exp4"]["e"].append(float(m.group(2)))

with open("../results/leapfrog/leapfrog_cache.json", "w") as f:
    json.dump(data, f)
print("Data successfully cached to results/leapfrog/leapfrog_cache.json")