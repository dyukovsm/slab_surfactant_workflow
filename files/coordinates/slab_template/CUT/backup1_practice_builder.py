import subprocess
import random
from pathlib import Path

#base_path = Path("../../files/coordinates/slab_template/PME")
base_path = Path("./")
mdp_file = base_path / "PRO_SURFTEN.mdp"
tpr = base_path / "PRO_SURFTEN_CHUNK_1.tpr"
trr = base_path / "PRO_SURFTEN_CHUNK_1.trr"
output_gro = "random_frame.gro"

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

frame_interval = nstxout * dt
max_time = nsteps * dt
valid_times = [i * frame_interval for i in range(0, int(nsteps / nstxout) + 1)]
valid_times[-1] = max_time

random_time = random.choice(valid_times)
print(f"Extracting frame at t = {random_time:.3f} ps")

subprocess.run([
    "/usr/local/gromacs/bin/gmx", "trjconv",
    "-s", str(tpr),
    "-f", str(trr),
    "-o", output_gro,
    "-dump", str(random_time)
], input="0\n", text=True, check=True)
