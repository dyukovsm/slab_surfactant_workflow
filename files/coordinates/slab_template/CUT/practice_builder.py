import subprocess
import random
from pathlib import Path

# === USER CONFIG ===
pick_randomTrue_pick_allFalse = False  # True -> pick random frame; False -> dump all frames separately

# === FILE PATHS ===
base_path = Path("../../files/coordinates/slab_template/CUT")
#base_path = Path("./")
mdp_file = base_path / "PRO_SURFTEN.mdp"
tpr = base_path / "PRO_SURFTEN_CHUNK_1.tpr"
trr = base_path / "PRO_SURFTEN_CHUNK_1.trr"
output_gro = "random_frame.gro"

# === PARSE MDP PARAMETERS (fast method) ===
params = {}
with open(mdp_file) as f:
    for line in f:
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().split()[0]  # first token after '='
        if key in {"dt", "nsteps", "nstxout", "nstvout"}:
            params[key] = float(val)

dt = params["dt"]
nsteps = int(params["nsteps"])
nstxout = int(params["nstxout"])

# === COMPUTE FRAME TIMES ===
frame_interval = nstxout * dt
max_time = nsteps * dt
valid_times = [i * frame_interval for i in range(0, int(nsteps / nstxout) + 1)]
valid_times[-1] = max_time  # ensure exact last time

# === EXECUTION ===
if pick_randomTrue_pick_allFalse:
    random_time = random.choice(valid_times)
    print(f"Extracting random frame at t = {random_time:.3f} ps")
    subprocess.run([
        "/usr/local/gromacs/bin/gmx", "trjconv",
        "-s", str(tpr),
        "-f", str(trr),
        "-o", output_gro,
        "-dump", str(random_time)
    ], input="0\n", text=True, check=True)
else:
    print(f"Extracting all {len(valid_times)} frames individually...")
    for i, t in enumerate(valid_times):
        out_name = f"frame_{i:05d}.gro"
        print(f"  → dumping t = {t:.3f} ps to {out_name}")
        subprocess.run([
            "/usr/local/gromacs/bin/gmx", "trjconv",
            "-s", str(tpr),
            "-f", str(trr),
            "-o", out_name,
            "-dump", str(t)
        ], input="0\n", text=True, check=True)

