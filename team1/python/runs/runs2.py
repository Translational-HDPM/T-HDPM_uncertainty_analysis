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
    file.write('module load anaconda \n')
    file.write('conda activate Monte-Carlo \n')
    cmd = 'python Data-Centric-Sim.py -filter {} -col {} -by {} -scaler {} -folder Final-Sprint-DataDriven \n'.format(vals,cols,method,scaler)
    file.write(cmd)
    file.close()
    os.system('sbatch job.submit')
    os.system('rm job.submit')


for vals in [0,100,200,400,800]:
    for cols in ['Coeff-AVG','Coeff-MED','AVG','MED']:
        for method in ['smallest']:
            for scaler in ['StandardScaler']:
                runjob(vals,cols,method,scaler)

                