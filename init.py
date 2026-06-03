import signac
import os

project = signac.init_project()

# Statepoint parameters
surfact_count = [0]
statepoint_bkup_0 = [0]
statepoint_bkup_1 = [0]
statepoint_bkup_2 = [0]
statepoint_bkup_3 = [0]

# Set up statepoints
total_statepoints = []

for sc in surfact_count:
    for b0 in statepoint_bkup_0:
        for b1 in statepoint_bkup_1:
            for b2 in statepoint_bkup_2:
                for b3 in statepoint_bkup_3:
                    statepoint = {
                        "surfact_count": sc,
                        "statepoint_bkup_0": b0,
                        "statepoint_bkup_1": b1,
                        "statepoint_bkup_2": b2,
                        "statepoint_bkup_3": b3
                    }
                    total_statepoints.append(statepoint)


with open('legend.txt', 'w') as legend:
    legend.write('job_id \t\t\t\t\t\t statepoint\n')
    print('job_id \t\t\t\t\t\t statepoint')
    
    for sp in total_statepoints:
        job = project.open_job(statepoint=sp).init()
        legend.write(f'{job} \t {sp}\n')
        print(f'{job} \t {sp}')

