# Surfactant on Water Slab Simulation Workflow

This repository provides a signac-flow workflow designed to run slab simulations of surfactants on water surfaces. It was initialized from the water workflow in `LJ-PME_sharedWorkflow` and has been cleaned up and renamed for customization.


## Setup

Ensure your conda/mamba environment has signac-flow, mbuild, foyer, and GMSO installed. You can activate your environment and initialize the workflow as follows.

```bash
# Initialize the signac project
python init.py

# Check status of the workflow
python project.py status
```

## Workflow Architecture

* `init.py` — Defines the statepoints (r_cut, cut_type, temperature, replicas) and initializes the Signac project.
* `project.py` — Defines the workflow steps (operations and preconditions/postconditions).
* `files/`
  * `coordinates/` — Starting structural files (`SPCE.mol2`, etc.).
  * `xml/water/` — Force field XML files for parametrization (`SPCE_GMSO.xml`).
  * `mdp/` — Template MDP files for GROMACS.
  * `python_files/` — Helper python modules:
    * `names.py` — System configurations, files, and chunking dictionaries.
    * `job_tester.py` — Precondition and postcondition functions for signac operations.
    * `job_templates.py` — MDP writing and slab builder helper functions.
* `templates/`
  * `gmx_grid_fall2025.sh` — Slurm submission template.

## Modifying for Surfactants

To run surfactant-on-water simulations using typing schemas like OpenFF:
1. Revise the `BUILD_INPUT` operation in `project.py` to construct the mixed surfactant-water system using `mbuild`.
2. Parameterize the water and surfactant molecules (e.g. using `foyer`, `gmso`, or `openff`).
3. Set up the topology and coordinate files (`init.top`, `init.gro`) accordingly.
