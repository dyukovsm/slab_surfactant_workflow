import math
import mbuild as mb
from foyer import Forcefield
import foyer
import pandas as pd
import numpy as np
import random
import time
from parmed import residue
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import parmed
from parmed import gromacs
import signac
from flow import FlowProject, aggregator
from flow.environment import DefaultSlurmEnvironment
import flow
import subprocess
import os
from jinja2 import Environment, FileSystemLoader
import shutil
from files.python_files import names, job_tester
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


GMX_PREFIX = names.GMX_PREFIX

def simple_mdp_writer(job,mdp_name,parameters,constraints=None,templates_dir=None,template_name=None):
    loader = FileSystemLoader('.')
    env = Environment(loader=loader)
    path = os.path.relpath(f'{templates_dir}')
    MDP_NAME = template_name
    
    if constraints is None:
        update_dict = {
            'constraints_string' : ';',
            'constraints' : 'whatever',
            'constraint_algorithm_string' : ';',
            'constraint_algorithm' : 'whatever',
            'lincs_order_string' : ';',
            'lincs_order' : 'whatever'
        }
    elif 'lincs' in constraints:
        update_dict = {
            'constraints_string' : 'constraints         = ',
            'constraints' : 'all-bonds',
            'constraint_algorithm_string' : 'constraint-algorithm = ',
            'constraint_algorithm' : 'LINCS',
            'lincs_order_string' : 'lincs-order           = ',
            'lincs_order' : '6'
        }
    elif 'shake' in constraints:
        update_dict = {
            'constraints_string' : 'constraints         = ',
            'constraints' : 'all-angles',
            'constraint_algorithm_string' : 'constraint-algorithm = ',
            'constraint_algorithm' : 'SHAKE',
            'lincs_order_string' : 'shake-tol           = ',
            'lincs_order' : '0.00001'
        }
    parameters.update(update_dict)
    
    template_data = parameters
    template = env.get_template(f'{path}/{MDP_NAME}')
    
    output = template.render(template_data)
    with open(f'workspace/{job}/{mdp_name}','w') as f:
        f.write(output)
        
    
def gimme_dir(job):
    current_dir = os.getcwd()
    job_dir = f'{current_dir}/workspace/{job}' 
    return current_dir, job


def write_gmxINDEX_forRESIDUES(job, top_file = 'init.top', gro_file = 'init.gro', index_file_name = 'whacky_index_file.ndx'):

    #system_pmdTop = gromacs.GromacsTopologyFile('init.top')
    #gmx_gro = gromacs.GromacsGroFile.parse(f'init.gro')
    with(job):
        system_pmdTop = gromacs.GromacsTopologyFile(f'{top_file}')
        gmx_gro = gromacs.GromacsGroFile.parse(f'{gro_file}')
        system_pmdTop.box = gmx_gro.box
        system_pmdTop.positions = gmx_gro.positions

        angles4Gromacs = open(f'{index_file_name}','w')
        angles4Gromacs.write('[ WAT ] ;index1, atom_type\n')
        some_angles_written = False
        for i in system_pmdTop.residues:
            comments = [] 
            for j in i.atoms:
                correct_index = j.idx + 1
                comments.append(j)
                angles4Gromacs.write(f'{correct_index}\t')
                some_angles_written = True
            angles4Gromacs.write(f' \t;\t{str(comments)}\n')

        if some_angles_written:
            angles4Gromacs.write('\n;index file written correctly \n')
        angles4Gromacs.close() 
        
        
def manual_gmx_index_file_make(job,gro_file = 'init.gro', index_file_name = 'whacky_index_file.ndx',skip_residues_from_ncompounds=1000):
    with(job):
        skip_guess = skip_residues_from_ncompounds
        #skip_guess = math.ceil(math.log10(skip_guess))
        skip_guess = len(str(skip_guess))

        # Open the file and read the third line
        with open(f"{gro_file}", 'r') as f:
            for _ in range(2):
                next(f)  # skip first two lines
            line = f.readline()

        # Determine the column widths
        end_positions = [i for i, char in enumerate(line) if char != ' ' and (i == len(line) - 1 or line[i+1] == ' ')]
        column_widths = [end_positions[0] + 1] + [end_positions[i] - end_positions[i-1] for i in range(1, len(end_positions))]

        # Use numpy's genfromtxt to read the data with the determined column widths
        data = np.genfromtxt(f"{gro_file}", dtype=None, skip_header=2, delimiter=column_widths, encoding='utf-8')

        data=data[:-1]
        print(data)
        index_column = data['f2']

        result_dict = defaultdict(list)


        for record in data:
            #if record['f0'] == 
            key = record['f0'].strip()
            value = record['f2']#.strip()

            result_dict[key].append(value)

        index_file = open(f'{index_file_name}','w')
        header_preper = record['f0'].strip()#[0]
        header_preper = header_preper[skip_guess:-1]
        index_file.write(f'[ {header_preper} ]\n')

        #print('________________________________________________')
        #for i in range(10):
        #    print(header_preper)

        #sprint('________________________________________________')

        for i in result_dict.keys():
            #print(result_dict[i]) 
            dummy_list = result_dict[i]
            for j in dummy_list:
                index_file.write(f'{j}\t')
            index_file.write(f'\t ; {i} \n')

        index_file.close()
        
        
def gmx_density_profile(job, trr_or_gro, index_file, tpr_file, output_xvg_name, first_frame, last_frame, slices=128):
    'return a profile of density. remember to use gpu node if tpr was also made on gpu'
    with(job):
        #comman_string = str(f'{GMX_PREFIX}') + str(' -f density ') + str(f'{trr_or_gro}') + str(' -n ') + str(f'{index_file}') + str(' -s ') + str(f'{tpr_file}') + str(' -o ') + str(f'{output_xvg_name}') + str(' -sl ')# + str(f'{slices})
        #comman_string = comman_string + str(f'{slices})
        #subprocess.run(comman=str(f'{GMX_PREFIX}') + str(' -f density ') + f'{trr_or_gro}' + ' -n ' + f'{index_file}' + ' -s ' + f'{tpr_file}', ' -o ', f'{output_xvg_name}', ' -sl ', f'{slices}'),shell=True)
        #f"{var} {var} text {var}"
        subprocess.run((f'{GMX_PREFIX}') + str(' density -f ') + str(f'{trr_or_gro}') + str(' -n ') + str(f'{index_file}') + str(' -s ') + str(f'{tpr_file}') + str(' -o ') + str(f'{output_xvg_name}') + str(' -sl ') + str(f'{slices}'),shell=True)
        #subprocess.run()#str(f'{GMX_PREFIX}') + str(' -f density ') + f'{trr_or_gro}' + ' -n ' + f'{index_file}' + ' -s ' + f'{tpr_file}', ' -o ', f'{output_xvg_name}', ' -sl ', f'{slices}'),shell=True)

        #p = subprocess.Popen([f'{GMX_PREFIX}', '-f', 'density', f'{trr_or_gro}', f'-n', f'{index_file}', f'-s', f'{tpr_file}', '-o', f'{output_xvg_name}', '-sl', f'{slices}'], stdin=subprocess.PIPE,stdout=subprocess.PIPE, cwd=os.getcwd())
        #out,err = p.communicate(input=str_sorrounding)
        #capture = p.decode() 
        

## def give_name_return_whichChunk(job, chunk_dict):
##     with(job):
##         last_chunk = 0
##         for key in chunk_dict.keys():
##             keys_to_list = list(chunk_dict.keys())
##             #print(f'last_chunk : {last_chunk}')
##             working_key = min(key+int(1), keys_to_list[-1])
##             
##             input_log_file = f'{chunk_dict[working_key]}.log'
##             if os.path.isfile(input_log_file):
##                 if job_tester.look_in_file(job, [input_log_file], "Finished",check_for_not=True,check_for_not_str='Received the TERM'):
##                     last_chunk = last_chunk + 1
##             else:
##                 break
##                 
##         return last_chunk
    
    
def give_name_return_whichChunk(job, chunk_dict):
    with(job):
        last_chunk = 0
        for key in chunk_dict.keys():
            #keys_to_list = list(chunk_dict.keys())
            print(f'last_chunk : {last_chunk}')
            #working_key = max(key+1, keys_to_list[-1])
            working_key = key+1
            input_log_file = f'{chunk_dict[working_key]}.log'
            if os.path.isfile(input_log_file):
                if job_tester.look_in_file(job, [input_log_file], "Finished",check_for_not=True,check_for_not_str='Received the TERM'):
                    last_chunk = last_chunk + 1
                    # last_chunk = min(last_chunk + 1,keys_to_list[-1])
                else:
                    break
            else:
                break
            
                
        return last_chunk
    

def build_slab_from_template(job, template_file_trr, template_file_tpr, template_file_mdp, 
                             output_name, pick_randomTrue_pick_allFalse=True):
        #with(job):
        
        # === FILE PATHS ===
        #base_path = Path("../../files/coordinates/slab_template/CUT")
        #base_path = Path("./")
        mdp_file = f"{template_file_mdp}" # mdp_file = base_path / "PRO_SURFTEN.mdp"
        tpr = f"{template_file_tpr}" # tpr = base_path / "PRO_SURFTEN_CHUNK_1.tpr"
        trr = f"{template_file_trr}" # trr = base_path / "PRO_SURFTEN_CHUNK_1.trr"
        output_gro = f"{output_name}" # output_gro = "random_frame.gro"

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
                f"{GMX_PREFIX}", "trjconv",
                "-s", str(tpr),
                "-f", str(trr),
                "-o", output_gro,
                "-dump", str(random_time)
            ], input="0\n", text=True, check=True)
        else:
            print(f"Extracting all {len(valid_times)} frames individually...")
            for i, t in enumerate(valid_times):
                out_name = f"{output_gro}_frame_{i:05d}.gro"
                print(f"  → dumping t = {t:.3f} ps to {out_name}")
                subprocess.run([
                    f"{GMX_PREFIX}", "trjconv",
                    "-s", str(tpr),
                    "-f", str(trr),
                    "-o", out_name,
                    "-dump", str(t)
                ], input="0\n", text=True, check=True)
