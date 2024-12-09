import os

def runjob(u,vals,cols,method,scaler,dists):
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
    cmd = 'python run-MC-sim.py -int {} -filter {} -col {} -by {} -scaler {} -dist {} -folder Final-Sprint \n'.format(u,vals,cols,method,scaler,dists)
    file.write(cmd)
    file.close()
    os.system('sbatch job.submit')
    os.system('rm job.submit')

for vals in [100,200,400,800]:
    for cols in ['Coeff-TPM']:
        for method in ['smallest']:
            for scaler in ['StandardScaler']:
                for dists in ['normal','LHS']:
                    for us in ['.1','.2','.3','.5']:
                        runjob(us,vals,cols,method,scaler,dists)
                    