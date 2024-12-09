import os

def runjob(vals,cols,method,scaler):
    file = open('job.submit','w')
    file.write('#!/bin/sh \n')
    file.write('#SBATCH --nodes=1 \n')
    file.write('#SBATCH --ntasks=1  \n')
    file.write('#SBATCH --time=3:30:00 \n')
    file.write('#SBATCH --cpus-per-task=128 \n')
    file.write('#SBATCH --mem=100G \n')
    file.write('#SBATCH --job-name no-filter \n')
    file.write('#SBATCH --job-name no-filter \n')
    file.write('#SBATCH --job-name no-filter \n')
    file.write('conda activate Monte-Carlo \n')
    cmd = 'python Data-Centric-Sim.py -filter {} -col {} -by {} -scaler {} -folder Team-3-DD-Scores \n'.format(vals,cols,method,scaler)
    file.write(cmd)
    file.close()
    os.system('sbatch job.submit')
    os.system('rm job.submit')


for vals in [0]:
    for cols in ['Coeff-AVG']:
        for method in ['smallest']:
            for scaler in ['StandardScaler']:
                runjob(vals,cols,method,scaler)

                