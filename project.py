import math
import mbuild as mb
import numpy as np
import signac
from flow import FlowProject
from flow.environment import DefaultSlurmEnvironment
import os
import shutil
from files.python_files import names, job_tester, job_templates
import forcefield_utilities
import mbuild as mb
import gmso
from gmso.external.convert_mbuild import from_mbuild
from gmso.parameterization import apply
from gmso.formats.top import write_top
from gmso.formats.gro import write_gro
import matplotlib.pyplot as plt
import pandas as pd
import re
import subprocess
import io

PROJECT_FILES_DIR = os.path.abspath('files')
#/wsu/home/go/go07/go0719/GROMACS/2025/fall/methane_VLE/paper_template/files
#/wsu/home/go/go07/go0719/GROMACS/2025/fall/methane_VLE/paper_template/LJ-PME_sharedWorkflow/flows_of_work/water/
PROJECT_DIR = os.path.abspath('.')
MDP_DIR = 'mdp'; XYZ_DIR = 'coordinates'; XML_DIR = 'xml/water'

MIN_CORES = 1; BUILD_CORES = 2; MAX_CORES = 4
TINNY_MEM = 0.512; LOW_MEM = 1.024; HIGH_MEM = 2.048; MAX_MEM = 4.096
SHORT_WAIT = 2.0; HALF_DAY = 8.0; DAY_WAIT = 24.0; MED_WAIT = 96.0; ONE_WORKWEEK = 111.0; TWO_WEEKS = 123.0

SIMULATION_GPU = 1

PRINT_MY_NODE = 'echo -e "Hello World\nHello World upcomming hostname"; hostname'
WATER_STANDARD_RENAME = 'WAT'


# Global user-configurable simulation parameters
TEMPERATURE         = 400           # Target temperature in K
R_CUT               = 3.5           # Cut-off distance in nm
CUT_TYPE            = 'Cut-off'     # Always use Cut-off

# Steps count for production and equilibration (can be adjusted for testing vs production)
MID_EQ_STEPS        = int(1000)     # testing steps (production: 2000000)
LONG_EQ_STEPS       = int(1000)     # testing steps (production: 20000000)
SLOW_OUTPUT         = int(100)      # testing steps (production: 10000)
SLOW_CALC           = int(100)      

PRO_STEPS           = int(1000)     # testing steps (production: 10000000)
FAST_OUTPUT         = int(100)      # testing steps (production: 5000)
FAST_CALC           = int(100)      

EQ_CHUNK_COUNT = names.NAME_EQ_CHUNK_COUNT
PRO_CHUNK_COUNT = names.NAME_PRO_CHUNK_COUNT

CUT_CUBELENGTH_Z = 14.0
CUT_CUBELENGTH_XY = 8.1
CUT_CUBELENGTH_Z_FINAL = 50.0

N_MOLECULES_CUT = int(27500) 
GMX_PREFIX = names.GMX_PREFIX

current_directory = os.getcwd()
current_directory_name = os.path.basename(current_directory)
project = signac.get_project()

class Custom_environment(DefaultSlurmEnvironment):  
    hostname_pattern = r".*\.grid\.wayne\.edu"
    template = "gmx_grid_fall2025.sh"
    
###################################################################################################

@FlowProject.post(job_tester.mdps_written)
@FlowProject.operation(directives={ "np": BUILD_CORES,  "ngpu": 0, "memory": MAX_MEM, "walltime": SHORT_WAIT})
def BUILD_INPUT(job):
    with job:
        # Load water coordinates and set rigid molecule properties
        water = mb.load(f'{PROJECT_FILES_DIR}/{XYZ_DIR}/SPCE.mol2')
        water.name = 'WAT'
        
        starting_box = mb.fill_box(compound=water, n_compounds=N_MOLECULES_CUT, box=[CUT_CUBELENGTH_XY, CUT_CUBELENGTH_XY, CUT_CUBELENGTH_Z])
        wat_ff_xml = forcefield_utilities.GMSOFFs().load_xml(f'{PROJECT_FILES_DIR}/{XML_DIR}/SPCE_GMSO.xml').to_gmso_ff()
        
        gmso_starting_box = from_mbuild(starting_box)
        
        for site in gmso_starting_box.sites:
            if 'WAT' in site.molecule.name:
                site.label = WATER_STANDARD_RENAME
                site.molecule.isrigid = True
                site.molecule.name = WATER_STANDARD_RENAME
        
        print('Applying force field using GMSO...')
        apply(top=gmso_starting_box, forcefields=wat_ff_xml, identify_connections=True)

        write_gro(gmso_starting_box, filename='init.gro')
        write_top(gmso_starting_box, filename='init.top', settles_tag="WAT")
    
    chunked_eq = int(LONG_EQ_STEPS/EQ_CHUNK_COUNT)
    chunked_pro = int(PRO_STEPS/PRO_CHUNK_COUNT)
    
    # 1. TEMP_RAMP_START
    parameters = {
        'integrator' : 'md',
        'nsteps' : MID_EQ_STEPS,
        'output_control' : SLOW_OUTPUT,
        'nstcalcenergy' : FAST_CALC,
        'nstlist' : 10,
        'rcoulomb' : R_CUT,
        'coulombtype' : 'PME',
        'coulomb_modifier' : 'None',
        'rcoulomb_switch' : 0.0,
        'vdw_type' : CUT_TYPE,
        'vdw_modifier' : 'None',
        'rvdw' : R_CUT,
        'rvdw_switch' : 0.0,
        'DispCorr' : 'No',
        'tcouple' : 'nose-hoover',
        'ref_t' : 350 + 1/3*(TEMPERATURE - 350)
    }
    job_templates.simple_mdp_writer(job, mdp_name=f'{names.NAME_TEMP_RAMP_START}.mdp', parameters=parameters, constraints=None, templates_dir=f'{PROJECT_FILES_DIR}/mdp/', template_name='NVT_template_generic.mdp')

    # 2. TEMP_RAMP_STOP
    parameters.update({
        'nsteps' : MID_EQ_STEPS,
        'output_control' : SLOW_OUTPUT,
        'nstcalcenergy' : FAST_CALC,
        'ref_t' : 350 + 2/3*(TEMPERATURE - 350)
    })
    job_templates.simple_mdp_writer(job, mdp_name=f'{names.NAME_TEMP_RAMP_STOP}.mdp', parameters=parameters, constraints=None, templates_dir=f'{PROJECT_FILES_DIR}/mdp/', template_name='NVT_template_generic.mdp')

    # 3. EQ_SURFTEN
    parameters.update({
        'nsteps' : LONG_EQ_STEPS,
        'output_control' : SLOW_OUTPUT,
        'nstcalcenergy' : FAST_CALC,
        'ref_t' : TEMPERATURE 
    })
    job_templates.simple_mdp_writer(job, mdp_name=f'{names.NAME_EQ_SURFTEN}.mdp', parameters=parameters, constraints=None, templates_dir=f'{PROJECT_FILES_DIR}/mdp/', template_name='NVT_template_generic.mdp')

    # 4. PRO_SURFTEN
    parameters.update({
        'nsteps' : chunked_pro,
        'output_control' : FAST_OUTPUT,
        'nstcalcenergy' : FAST_CALC,
        'ref_t' : TEMPERATURE 
    })
    job_templates.simple_mdp_writer(job, mdp_name=f'{names.NAME_PRO_SURFTEN}.mdp', parameters=parameters, constraints=None, templates_dir=f'{PROJECT_FILES_DIR}/mdp/', template_name='NVT_template_generic.mdp')


@FlowProject.pre(job_tester.mdps_written)
@FlowProject.post(job_tester.inits_written)
@FlowProject.post(job_tester.build_surfTen_nvt_done)
@FlowProject.operation(directives={ "np": BUILD_CORES,  "ngpu": 1, "memory": MAX_MEM, "walltime": SHORT_WAIT})
def BUILD_INPUT_FROM_TEMPLATE(job):
    with(job):
        template_dir = f'{PROJECT_FILES_DIR}/{XYZ_DIR}/slab_template/CUT'

        shutil.copy(f'{template_dir}/{names.NAME_INPUT_TEMPLATE_SLAB}.trr', '.')
        shutil.copy(f'{template_dir}/{names.NAME_INPUT_TEMPLATE_SLAB}.tpr', '.')
        shutil.copy(f'{template_dir}/{names.NAME_INPUT_TEMPLATE_SLAB}.mdp', '.')
        shutil.copy(f'{template_dir}/init.top', '.')

        job_templates.build_slab_from_template(job, template_file_trr=f'{names.NAME_INPUT_TEMPLATE_SLAB}.trr', template_file_tpr=f'{names.NAME_INPUT_TEMPLATE_SLAB}.tpr',
                                               template_file_mdp=f'{names.NAME_INPUT_TEMPLATE_SLAB}.mdp', output_name=f'{names.NAME_ELONGATED}.gro',pick_randomTrue_pick_allFalse=True)

        

###################################################################################################
###################################################################################################
#### need to load the template gro surften file (different files for pme vs cut) save as init.gro
###################################################################################################
###################################################################################################


@FlowProject.pre(job_tester.inits_written)
@FlowProject.pre(job_tester.important_jobs)
@FlowProject.pre(job_tester.build_surfTen_nvt_done)
@FlowProject.post(job_tester.temp_ramp_start_done)
@FlowProject.operation(directives={ "np": MAX_CORES,  "ngpu": SIMULATION_GPU, "memory": HIGH_MEM, "walltime": TWO_WEEKS},with_job=True,cmd=True)
def TEMP_RAMP_START(job):
    
    #last_completed_chunk = job_templates.give_name_return_whichChunk(job,names.EQ_SURFTEN_CHUNK_TO_STARTING_GRO_FILE)
    input_file = names.NAME_ELONGATED #names.EQ_SURFTEN_CHUNK_TO_STARTING_GRO_FILE[last_completed_chunk]
    output_file = names.NAME_TEMP_RAMP_START #names.EQ_SURFTEN_CHUNK_TO_STARTING_GRO_FILE[last_completed_chunk+1]
    
    build_mdp = str(GMX_PREFIX + ' grompp -f ' + f'{output_file}.mdp -c ' + f'{input_file}.gro -p ' + 'init.top -o ' + f'{output_file}.tpr -maxwarn 999')
    #build_mdp = str(GMX_PREFIX + ' grompp -f ' + f'{names.NAME_EQ_SURFTEN}.mdp -c ' + f'last_frame.gro -p ' + 'init.top -o ' + f'{names.NAME_EQ_SURFTEN}.tpr -maxwarn 999999')
    run_mdp = str(GMX_PREFIX + f' mdrun -nt ' + f'{MAX_CORES}' + ' -deffnm' + f' {output_file}')    # + ' -nb cpu')# + ' -pin on')
    run_command = str(PRINT_MY_NODE + '; ' + 'sleep 2' + '; ' + build_mdp + '; ' + 'sleep 2' + '; ' + run_mdp)
    
    return run_command

###################################################################################################

@FlowProject.pre(job_tester.important_jobs)
@FlowProject.pre(job_tester.temp_ramp_start_done)
@FlowProject.post(job_tester.temp_ramp_stop_done)
@FlowProject.operation(directives={ "np": MAX_CORES,  "ngpu": SIMULATION_GPU, "memory": HIGH_MEM, "walltime": TWO_WEEKS},with_job=True,cmd=True)
def TEMP_RAMP_STOP(job):
    
    #last_completed_chunk = job_templates.give_name_return_whichChunk(job,names.EQ_SURFTEN_CHUNK_TO_STARTING_GRO_FILE)
    input_file = names.NAME_TEMP_RAMP_START #names.EQ_SURFTEN_CHUNK_TO_STARTING_GRO_FILE[last_completed_chunk]
    output_file = names.NAME_TEMP_RAMP_STOP #names.EQ_SURFTEN_CHUNK_TO_STARTING_GRO_FILE[last_completed_chunk+1]
    
    build_mdp = str(GMX_PREFIX + ' grompp -f ' + f'{output_file}.mdp -c ' + f'{input_file}.gro -p ' + 'init.top -o ' + f'{output_file}.tpr -maxwarn 999')
    #build_mdp = str(GMX_PREFIX + ' grompp -f ' + f'{names.NAME_EQ_SURFTEN}.mdp -c ' + f'last_frame.gro -p ' + 'init.top -o ' + f'{names.NAME_EQ_SURFTEN}.tpr -maxwarn 999999')
    run_mdp = str(GMX_PREFIX + f' mdrun -nt ' + f'{MAX_CORES}' + ' -deffnm' + f' {output_file}')    # + ' -nb cpu')# + ' -pin on')
    run_command = str(PRINT_MY_NODE + '; ' + 'sleep 2' + '; ' + build_mdp + '; ' + 'sleep 2' + '; ' + run_mdp)
    
    return run_command

###################################################################################################

@FlowProject.pre(job_tester.important_jobs)
@FlowProject.pre(job_tester.temp_ramp_stop_done)
@FlowProject.post(job_tester.eq_nvt_surften_done)
@FlowProject.operation(directives={ "np": MAX_CORES,  "ngpu": SIMULATION_GPU, "memory": HIGH_MEM, "walltime": TWO_WEEKS},with_job=True,cmd=True)
def EQ_SURFTEN(job):
    
    last_completed_chunk = job_templates.give_name_return_whichChunk(job,names.EQ_SURFTEN_CHUNK_TO_STARTING_GRO_FILE)
    input_file = names.NAME_TEMP_RAMP_STOP #names.EQ_SURFTEN_CHUNK_TO_STARTING_GRO_FILE[last_completed_chunk]
    output_file = names.EQ_SURFTEN_CHUNK_TO_STARTING_GRO_FILE[last_completed_chunk+1]
    
    build_mdp = str(GMX_PREFIX + ' grompp -f ' + f'{names.NAME_EQ_SURFTEN}.mdp -c ' + f'{input_file}.gro -p ' + 'init.top -o ' + f'{output_file}.tpr -maxwarn 999')
    #build_mdp = str(GMX_PREFIX + ' grompp -f ' + f'{names.NAME_EQ_SURFTEN}.mdp -c ' + f'last_frame.gro -p ' + 'init.top -o ' + f'{names.NAME_EQ_SURFTEN}.tpr -maxwarn 999999')
    run_mdp = str(GMX_PREFIX + f' mdrun -nt ' + f'{MAX_CORES}' + ' -deffnm' + f' {output_file}')    # + ' -nb cpu')# + ' -pin on')
    run_command = str(PRINT_MY_NODE + '; ' + 'sleep 2' + '; ' + build_mdp + '; ' + 'sleep 2' + '; ' + run_mdp)
    
    return run_command

###################################################################################################

@FlowProject.pre(job_tester.important_jobs)
@FlowProject.pre(job_tester.eq_nvt_surften_done)
@FlowProject.post(job_tester.pro_nvt_surften_done)
@FlowProject.operation(directives={ "np": MAX_CORES,  "ngpu": SIMULATION_GPU, "memory": HIGH_MEM, "walltime": TWO_WEEKS},with_job=True,cmd=True)
def PRO_SURFTEN(job):
    
    last_completed_chunk = job_templates.give_name_return_whichChunk(job,names.PRO_SURFTEN_CHUNK_TO_STARTING_GRO_FILE)
    #keys_to_list = list(chunk_dict.keys())
    
    #for i in range(10):
    #    print(f'last_completed_chunk : {last_completed_chunk}')
    print(f'last_completed_chunk: {last_completed_chunk}')
    input_file = names.PRO_SURFTEN_CHUNK_TO_STARTING_GRO_FILE[last_completed_chunk]
    output_file = names.PRO_SURFTEN_CHUNK_TO_STARTING_GRO_FILE[last_completed_chunk+1]
    
    build_mdp = str(GMX_PREFIX + ' grompp -f ' + f'{names.NAME_PRO_SURFTEN}.mdp -c ' + f'{input_file}.gro -p ' + 'init.top -o ' + f'{output_file}.tpr -maxwarn 999')
    run_mdp = str(GMX_PREFIX + f' mdrun -nt ' + f'{MAX_CORES}' + ' -deffnm' + f' {output_file}')    #+ ' -nb cpu')# + ' -pin on')
    run_command = str(PRINT_MY_NODE + '; ' + 'sleep 2' + '; ' + build_mdp + '; ' + 'sleep 2' + '; ' + run_mdp)
    
    return run_command


###################################################################################################

#@FlowProject.pre(job_tester.temp_ramp_stop_done)
@FlowProject.pre(job_tester.pro_nvt_surften_done)
@FlowProject.post(job_tester.data_collected)
@FlowProject.operation(directives={ "np": int(1),  "ngpu": SIMULATION_GPU, "memory": 1.1, "walltime": SHORT_WAIT})
def GRAPH_AND_COLLECT_PROPERTIES(job):
    with(job):
        last_completed_chunk = job_templates.give_name_return_whichChunk(job,names.PRO_SURFTEN_CHUNK_TO_STARTING_GRO_FILE)

        output_file = names.PRO_SURFTEN_CHUNK_TO_STARTING_GRO_FILE[last_completed_chunk+1]
        #output_file = f'{names.NAME_TEMP_RAMP_STOP}' # names.PRO_SURFTEN_CHUNK_TO_STARTING_GRO_FILE[last_completed_chunk+1]

        # try to match both this and the strings below with gromacs promp
        properties_of_interest = ["Potential", "LJ-(SR)", "Coulomb-(SR)", "Coul.-recip.", "Total-Energy", "Vir-ZZ", "Pres-ZZ", "#Surf*SurfTen"]

        
        # properties_of_interest_to_search_string_dict[properties_of_interest] has the exact strings to search for
        properties_of_interest_to_search_string_dict = {
            # ex:
            # Disper.-corr.           : ['Disper. corr.','(kJ/mol)'] 
            # Pres-ZZ                 : ['Pres-ZZ','(bar)']
            # Total-Energy            : ['Total Energy','(kJ/mol)']
            properties_of_interest[0] : ['Potential','(kJ/mol)'],
            properties_of_interest[1] : ['LJ (SR)','(kJ/mol)'],
            properties_of_interest[2] : ['Coulomb (SR)','(kJ/mol'],
            properties_of_interest[3] : ['Coul. recip.','(kJ/mol'],
            properties_of_interest[4] : ['Total Energy','(kJ/mol)'],
            properties_of_interest[5] : ['Vir-ZZ','(kJ/mol)'], # gotto love how gmx swaps them ;p
            properties_of_interest[6] : ['Pres-ZZ','(bar)'],
            properties_of_interest[7] : ['#Surf*SurfTen','(bar nm)']
        }
        
        properties_of_interest_storage_dict = {
            properties_of_interest[0] : 0.0 ,
            properties_of_interest[1] : 0.0 ,
            properties_of_interest[2] : 0.0 ,
            properties_of_interest[3] : 0.0 ,
            properties_of_interest[4] : 0.0 ,
            properties_of_interest[5] : 0.0 ,
            properties_of_interest[6] : 0.0,
            properties_of_interest[7] : 0.0
        }

        gromacs_input = b'1\n0\n'
        result = subprocess.run(
        [f"{names.GMX_PREFIX}", "energy", "-f", f"{output_file}.edr", "-o", "dummy_data.xvg"],
        input=gromacs_input,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT 
        )
        
        with open("gmx_energy_index_reader.txt", "wb") as f:
            f.write(result.stdout)
            
        with open("gmx_energy_index_reader.txt", "r") as f:
            text = f.read()
            
        pattern = r'(\d+)\s+(\S+)'

        matches = re.findall(pattern, text, re.MULTILINE)
        index_map = {}
        for index, name in matches:
            clean_name = name.strip()
            index_map[clean_name] = int(index)

        results = {}
        for prop in properties_of_interest:
            if prop in index_map:
                results[prop] = index_map[prop]
                print(f'index_map[prop] {index_map[prop]}')
            elif "Total" in index_map or "Total" in prop:
                print(f'index_map: {index_map} did not match with prop {prop}')
                
        
        newline_string = "\n".join(str(results[prop]) for prop in properties_of_interest if prop in results)
        
        initialBox =  mb.load(f'{output_file}.gro')
        boxLength = initialBox.box.lengths
        number_density_profile_bins = int(boxLength[2]*2.0)
        
        p = subprocess.Popen([f'{names.GMX_PREFIX}', 'density', '-f', f'{output_file}.trr', '-s', f'{output_file}.tpr', '-o', 'dens_profile.xvg','-sl',f'{number_density_profile_bins}'], stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        out,err = p.communicate(f'0\n0\n'.encode('utf-8'))#(b'13\n14\n15\n16\n17\n18\n19\n20\n0\n')
        capture = out.decode()
        
        read_density = open(f'dens_profile.xvg','r')
        write_density = open(f'{names.DENS_LOCAL_DATA}.txt','w')
        read_density_lines = read_density.readlines(); read_density.close()
        for line in read_density_lines:
            if line.startswith('@') or line.startswith('#'):
                pass
            else:
                write_density.write(line)
                
        write_density.close()
        read_density = np.loadtxt(f'{names.DENS_LOCAL_DATA}.txt')
        col2 = read_density[:, 1]
        
        gas_dens = col2.min()
        liq_dens = col2.max()
        
        aggregate_densFile = open(f"../../{names.DENS_GLOBAL_DATA}.txt",'a')
        aggregate_densFile.write(f"{job.id:<42} {job.sp.surfact_count:<8} {job.sp.statepoint_bkup_0:<8} {job.sp.statepoint_bkup_1:<8} {job.sp.statepoint_bkup_2:<8} {job.sp.statepoint_bkup_3:<8}"
                                    f" \t\t {liq_dens:<9}"
                                    f" \t\t {gas_dens:<9}"
                                   "\n")
        

        ###### --------- Ensure the indeces correspond to properteis we need  --------- ######
        
        p = subprocess.Popen([f'{names.GMX_PREFIX}', '-quiet', 'energy', '-f', f'{output_file}.edr', '-o', f'{names.GENERAL_LOCAL_DATA}_{output_file}.xvg'], stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        out,err = p.communicate(f'{newline_string}'.encode('utf-8'))#(b'13\n14\n15\n16\n17\n18\n19\n20\n0\n')
        capture = out.decode()
        
        Dummy_GMX_output = open(f'{names.GENERAL_LOCAL_DATA}_{output_file}.txt','w')
        Dummy_GMX_output.write(capture)
        Dummy_GMX_output.close()
        
        Dummy_GMX_output = open(f'{names.GENERAL_LOCAL_DATA}_{output_file}.txt','r')
        aggregate_surTenFile = open(f"../../{names.GENERAL_GLOBAL_DATA}.txt",'a')
        
                
        for a_single_line in Dummy_GMX_output:
            for property_str in properties_of_interest:
                search_property_str_dict = properties_of_interest_to_search_string_dict[property_str]
                search_str_start = search_property_str_dict[0]
                search_str_end = search_property_str_dict[1]
                
                if (search_str_start in a_single_line) and (search_str_end in a_single_line):
                    numpyCatcher=np.fromstring(a_single_line.strip(f'{search_str_start}{search_str_end}'),dtype=float,sep=' ')[0]
                    properties_of_interest_storage_dict[property_str] = numpyCatcher
                                    
            # read Dummy_GMX_output write to aggregate_surTenFile
            
        aggregate_surTenFile.write(f"{job.id:<42} {job.sp.surfact_count:<8} {job.sp.statepoint_bkup_0:<8} {job.sp.statepoint_bkup_1:<8} {job.sp.statepoint_bkup_2:<8} {job.sp.statepoint_bkup_3:<8}"
                                   f" {properties_of_interest_storage_dict[properties_of_interest[0]]:<42} " 
                                   f" {properties_of_interest_storage_dict[properties_of_interest[1]]:<42} " 
                                   f" {properties_of_interest_storage_dict[properties_of_interest[2]]:<42} " 
                                   f" {properties_of_interest_storage_dict[properties_of_interest[3]]:<42} "
                                   f" {properties_of_interest_storage_dict[properties_of_interest[4]]:<42} "
                                   f" {properties_of_interest_storage_dict[properties_of_interest[5]]:<42} "
                                   f" {properties_of_interest_storage_dict[properties_of_interest[6]]:<42} "
                                   f" {properties_of_interest_storage_dict[properties_of_interest[7]]:<42} "
                                   "\n")
        
        
        
        ### graph the data of simulation to see if it's converged

        Dummy_GMX_output.close(); aggregate_surTenFile.close()
        
        xvg_png_datasource = open(f'{names.GENERAL_LOCAL_DATA}_{output_file}.xvg','r')
        #lines = xvg_png_datasource.strip().split('\n')
        lines = xvg_png_datasource.readlines()
        
        header_lines = []
        data_lines = []
        in_data_section = False
        
        
        for line in lines:
            if line.startswith('@') or line.startswith('#'):
                header_lines.append(line)
                if "@TYPE xy" in line:
                    in_data_section = True
            else : # in_data_section # #and line.strip(): #and not line.startswith('#'):
                data_lines.append(line.strip())

        # Extract column names from header
        column_names = {}
        xaxis_label = "Time (ps)" # Default, will be updated if found
        yaxis_label = ""
        title = ""

        for line in header_lines:
            if line.startswith('@ s'):
                match = re.search(r'@ s(\d+) legend "(.+)"', line)
                if match:
                    col_index = int(match.group(1))
                    legend_name = match.group(2)
                    column_names[col_index] = legend_name
            elif line.startswith('@ xaxis'):
                match = re.search(r'@ xaxis\s+label "(.+)"', line)
                if match:
                    xaxis_label = match.group(1)
            elif line.startswith('@ yaxis'):
                match = re.search(r'@ yaxis\s+label "(.+)"', line)
                if match:
                    yaxis_label = match.group(1)
            elif line.startswith('@ title'):
                match = re.search(r'@\s+title "(.+)"', line)
                if match:
                    title = match.group(1)

        # Create a list of column names in order
        # The first column is always the x-axis label.
        # Then append the other column names based on their index.
        ordered_column_names = [xaxis_label]
        max_col_index = max(column_names.keys()) if column_names else -1
        for i in range(max_col_index + 1):
            if i in column_names:
                ordered_column_names.append(column_names[i])

        # Read data into a pandas DataFrame
        # Using io.StringIO to treat the list of data lines as a file
        #print(f'data_lines: {data_lines}')
        df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep=r'\s+', header=None)

        # Assign column names to the DataFrame
        df.columns = ordered_column_names[:len(df.columns)]

        # Plotting the data
        num_cols = len(df.columns) - 1  # Exclude the 'Time (ps)' column
        fig, axes = plt.subplots(num_cols, 1, figsize=(10, 5 * num_cols), sharex=True)

        if num_cols == 1:
            axes = [axes] # Ensure axes is iterable even for a single subplot
        
        for i, col_name in enumerate(df.columns[1:]): # Iterate over data columns, skipping time
            axes[i].plot(df[xaxis_label], df[col_name])
            axes[i].set_ylabel(f'{col_name} {yaxis_label}') # Add yaxis_label to each subplot's y-label
            axes[i].grid(True)
            #axes[i].set_title(f'{title}: {col_name}') # <- from chat
            key_to_mean_data = ''
            for key, value_list in properties_of_interest_to_search_string_dict.items(): 
                if col_name in value_list[0]:
                    key_to_mean_data = key
            if key_to_mean_data != '':
                axes[i].set_title(f'{col_name}; mean {properties_of_interest_storage_dict[key_to_mean_data]}') 

        axes[-1].set_xlabel(xaxis_label) # Set x-label only on the last subplot

        plt.tight_layout()
        plt.savefig(f'{names.GENERAL_LOCAL_DATA}_{output_file}.png')
        plt.close()

###################################################################################################


if __name__ == '__main__':
    FlowProject().main()
    
    
