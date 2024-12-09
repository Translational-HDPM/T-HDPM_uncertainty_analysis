import os

for i in [10,20,50,100,200,500,1000]:
    for j in [.01,.05,.10,.20]:
        os.system('python run-MC-sim.py -int .1 -filter 1 -col Avg-TPM -dist normal -folder MC-results-S4 -by IsolationForest -estimators {} -contamination {}'.format(i,j))
        os.system('python run-MC-sim.py -int .1 -filter 1 -col Avg-TPM -dist normal -folder MC-results-S4 -by IsolationForest -estimators {} -contamination {} --warm_start'.format(i,j))
        os.system('python run-MC-sim.py -int .1 -filter 1 -col Avg-TPM -dist normal -folder MC-results-S4 -by IsolationForest -estimators {} -contamination {} --bootstrap'.format(i,j))
        os.system('python run-MC-sim.py -int .1 -filter 1 -col Avg-TPM -dist normal -folder MC-results-S4 -by IsolationForest -estimators {} -contamination {} --warm_start --bootstrap'.format(i,j))
        
        os.system('python run-MC-sim.py -int .1 -filter 1 -col Coeff-TPM -dist normal -folder MC-results-S4 -by IsolationForest -estimators {} -contamination {}'.format(i,j))
        os.system('python run-MC-sim.py -int .1 -filter 1 -col Coeff-TPM -dist normal -folder MC-results-S4 -by IsolationForest -estimators {} -contamination {} --warm_start'.format(i,j))
        os.system('python run-MC-sim.py -int .1 -filter 1 -col Coeff-TPM -dist normal -folder MC-results-S4 -by IsolationForest -estimators {} -contamination {} --bootstrap'.format(i,j))
        os.system('python run-MC-sim.py -int .1 -filter 1 -col Coeff-TPM -dist normal -folder MC-results-S4 -by IsolationForest -estimators {} -contamination {} --warm_start --bootstrap'.format(i,j))
        