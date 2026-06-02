import signac
import os

project = signac.init_project()

r_cut = [0.8, 1.4, 2.0]
cut_type = ['Cut-off', 'PME'] 
temperature = [550]
replicas = [0] 

# Set up statepoints
total_statepoints = []

for rc in r_cut:
    for ct in cut_type:
        for rep in replicas:
            for temp in temperature:
                # Include Cut-off for all r_cut, but PME only for r_cut = 1.4
                if rc == 1.4 or "Cut-off" in ct:
                    statepoint = {
                        "r_cut": rc,
                        "cut_type": ct,
                        "replicas": rep,
                        "temperature": temp
                    }
                    total_statepoints.append(statepoint)

with open('legend.txt', 'w') as legend:
    legend.write('job_id \t\t\t\t\t\t statepoint\n')
    print('job_id \t\t\t\t\t\t statepoint')
    
    for sp in total_statepoints:
        job = project.open_job(statepoint=sp).init()
        legend.write(f'{job} \t {sp}\n')
        print(f'{job} \t {sp}')

