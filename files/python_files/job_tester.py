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
from files.python_files import names


extension_list_of_common_files = [".gro", ".trr", ".log", ".edr", ".tpr"]   # former extension_list_list
#extension_list_inits = [".gro",".top"]
extension_list_inits = [".top"]


def return_file_with_extensions(file_names,extension_list):
    file_names_with_extensions = [name + ext for name in file_names for ext in extension_list]

    return file_names_with_extensions


def test_existence_simple(job,file_lsit):
    with(job):
        test_passed = False
        for i in file_lsit:
            if job.isfile(i):
                test_passed = True
            elif not(job.isfile(i)):
                test_passed = False
                break
        return test_passed
    
    
## def look_in_file(job,file_list,look_string,debug=False,check_for_not=False,check_for_not_str=['']):
##     with(job):
##         test_passed = False
##         #if debug:
##         #    missing_file = open(f"debug_look_IN_file_{file_list[0]}.txt",'w')
##         #    missing_file.write('test')
##         #    missing_file.close()
##         #    #close the file before
##         #    missing_file = open(f"debug_look_IN_file_{file_list[0]}.txt",'a')
##         #    #for i in file_list:
##         #    #    missing_file.write(f'{i}\n')
##         for i in file_list:
##             if job.isfile(i):
##                 file_with_lines = open(f'{i}','r')
##                 lines = file_with_lines.readlines()
##                 for j in lines:
##                     #f debug:
##                     #   if look_string not in j:
##                     #       missing_file.write(f'{look_string} not found in \t\t\t {j}\n')
##                     #   else:
##                     #       missing_file.write(f'{look_string} WAS FOUND in \t\t\t {j}\n')
##                     if check_for_not:
##                         for single_string in check_for_not_str:
##                             if single_string in j:
##                                 #print(f'single_string : {single_string} for job {job.id}')
##                                 return test_passed
##                         
##                     if look_string in j:
##                         test_passed = True
##                         break
##                 file_with_lines.close()
##             #elif debug:
##             #    missing_file.write(f'ERROR {i} not found.\n')
##     return test_passed

def look_in_file(job,file_list,look_string,debug=False,check_for_not=False,check_for_not_str=['']):
    with(job):
        test_passed = False

        for i in file_list:
            if job.isfile(i):
                file_with_lines = open(f'{i}','r')
                lines = file_with_lines.readlines()
                for j in lines:

                    if check_for_not:
                        for single_string in check_for_not_str:
                            if single_string in j:

                                return test_passed
                        
                    if look_string in j:
                        test_passed = True
                        break
                file_with_lines.close()

    return test_passed

def run_only_one(job):
    test_passed = False
    if job.sp.replica < 1:
        test_passed = True
    return test_passed


def important_jobs(job):
    test_passed = False
    if job.sp.r_cut < 6.0:
        #if 'PME' in job.sp.cut_type:
            test_passed = True
    return test_passed

############################__JOB_SPECIFIC_FUNCTIONS__############################


@FlowProject.label
def inits_written(job):
    with(job):
        check_these_files = return_file_with_extensions(file_names=['init'],extension_list = extension_list_inits)
        return test_existence_simple(job,check_these_files)
    

@FlowProject.label
def mdps_written(job):
    with(job):
        check_these_files = return_file_with_extensions(file_names=[names.NAME_TEMP_RAMP_START, names.NAME_TEMP_RAMP_STOP,
                                                                    names.NAME_EQ_SURFTEN,
                                                                    names.NAME_PRO_SURFTEN],extension_list = ['.mdp'])
        return test_existence_simple(job,check_these_files)
    
    
@FlowProject.label
def build_input_starter(job):
    with(job):
        starter_bool = not(inits_written(job)) and not(mdps_written(job))
        return starter_bool
    

# equilibrated slabs copies.


##################################################################################

# TEMP_RAMP_START -> TEMP_RAMP_STOP -> EQ_SURFTEN -> PRO_SURFTEN
    
##################################################################################

@FlowProject.label
def build_surfTen_nvt_done(job):
    with(job):
        bool_job = False
        files_to_check = return_file_with_extensions(
            file_names=[f'{names.NAME_ELONGATED}'],
            extension_list=['.gro']
        )
        bool_job = test_existence_simple(job,files_to_check)
        return bool_job
    
##################################################################################

@FlowProject.label
def eq_nvt_post_em_files_present(job):
    with(job):
        bool_job = False
        files_to_check = return_file_with_extensions(
            file_names=[f'{names.NAME_TEMP_RAMP_START}'],
            extension_list=extension_list_of_common_files
        )
        bool_job = test_existence_simple(job,files_to_check)
        return bool_job
    
    
@FlowProject.label
def temp_ramp_start_done(job):
    with(job):
        bool_job = look_in_file(job,[f"{names.NAME_TEMP_RAMP_START}.log"],"Finished",check_for_not=True,check_for_not_str=['Received the TERM', 'Received the INT'])
        return bool_job
    
    
##################################################################################

@FlowProject.label
def temp_ramp_stop_done(job):
    with(job):
        bool_job = look_in_file(job,[f"{names.NAME_TEMP_RAMP_STOP}.log"],"Finished",check_for_not=True,check_for_not_str=['Received the TERM', 'Received the INT'])
        return bool_job
    
##################################################################################

@FlowProject.label
def build_surfTen_nvt_done(job):
    with(job):
        bool_job = False
        files_to_check = return_file_with_extensions(
            file_names=[f'{names.NAME_ELONGATED}'],
            extension_list=['.gro']
        )
        bool_job = test_existence_simple(job,files_to_check)
        return bool_job
    
##################################################################################


@FlowProject.label
def eq_nvt_surften_done(job):
    with(job):
        
        last_key_oneliner = list(names.EQ_SURFTEN_CHUNK_TO_STARTING_GRO_FILE.keys())[-1]
        bool_job = look_in_file(job,[f"{names.EQ_SURFTEN_CHUNK_TO_STARTING_GRO_FILE[last_key_oneliner]}.log"],"Finished",check_for_not=True,check_for_not_str=['Received the TERM', 'Received the INT'])
        return bool_job
        
          
##################################################################################


@FlowProject.label
def pro_nvt_surften_done(job):
    with(job):
        
        last_key_oneliner = list(names.PRO_SURFTEN_CHUNK_TO_STARTING_GRO_FILE.keys())[-1]
        bool_job = look_in_file(job,[f"{names.PRO_SURFTEN_CHUNK_TO_STARTING_GRO_FILE[last_key_oneliner]}.log"],"Finished",check_for_not=True,check_for_not_str=['Received the TERM', 'Received the INT'])
        return bool_job
    
    
##################################################################################


@FlowProject.label
def data_collected(job):
    test_passed = False
    local_name_of_file = f'{names.GENERAL_GLOBAL_DATA}.txt'
    if os.path.exists(local_name_of_file):
        with open(local_name_of_file, "r") as f:
            contents = f.read()
            if job.id in contents:
                test_passed = True
                
    return test_passed


