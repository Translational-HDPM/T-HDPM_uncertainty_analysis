#!/bin/bash

python run-MC-sim.py -int .1 -filter 1 -col Avg-TPM -dist normal -folder MC-results-S4 -by Grubbs -alpha .01
python run-MC-sim.py -int .1 -filter 1 -col Avg-TPM -dist normal -folder MC-results-S4 -by Grubbs -alpha .05
python run-MC-sim.py -int .1 -filter 1 -col Avg-TPM -dist normal -folder MC-results-S4 -by Grubbs -alpha .10
python run-MC-sim.py -int .1 -filter 1 -col Avg-TPM -dist normal -folder MC-results-S4 -by Grubbs -alpha .20

python run-MC-sim.py -int .1 -filter 1 -col Avg-TPM -dist normal -folder MC-results-S4 -by Grubbs-min -alpha .01
python run-MC-sim.py -int .1 -filter 1 -col Avg-TPM -dist normal -folder MC-results-S4 -by Grubbs-min -alpha .05
python run-MC-sim.py -int .1 -filter 1 -col Avg-TPM -dist normal -folder MC-results-S4 -by Grubbs-min -alpha .10
python run-MC-sim.py -int .1 -filter 1 -col Avg-TPM -dist normal -folder MC-results-S4 -by Grubbs-min -alpha .20

python run-MC-sim.py -int .1 -filter 1 -col Avg-TPM -dist normal -folder MC-results-S4 -by Grubbs-max -alpha .01
python run-MC-sim.py -int .1 -filter 1 -col Avg-TPM -dist normal -folder MC-results-S4 -by Grubbs-max -alpha .05
python run-MC-sim.py -int .1 -filter 1 -col Avg-TPM -dist normal -folder MC-results-S4 -by Grubbs-max -alpha .10
python run-MC-sim.py -int .1 -filter 1 -col Avg-TPM -dist normal -folder MC-results-S4 -by Grubbs-max -alpha .20


python run-MC-sim.py -int .1 -filter 1 -col Coeff-TPM -dist normal -folder MC-results-S4 -by Grubbs -alpha .01
python run-MC-sim.py -int .1 -filter 1 -col Coeff-TPM -dist normal -folder MC-results-S4 -by Grubbs -alpha .05
python run-MC-sim.py -int .1 -filter 1 -col Coeff-TPM -dist normal -folder MC-results-S4 -by Grubbs -alpha .10
python run-MC-sim.py -int .1 -filter 1 -col Coeff-TPM -dist normal -folder MC-results-S4 -by Grubbs -alpha .20

python run-MC-sim.py -int .1 -filter 1 -col Coeff-TPM -dist normal -folder MC-results-S4 -by Grubbs-min -alpha .01
python run-MC-sim.py -int .1 -filter 1 -col Coeff-TPM -dist normal -folder MC-results-S4 -by Grubbs-min -alpha .05
python run-MC-sim.py -int .1 -filter 1 -col Coeff-TPM -dist normal -folder MC-results-S4 -by Grubbs-min -alpha .10
python run-MC-sim.py -int .1 -filter 1 -col Coeff-TPM -dist normal -folder MC-results-S4 -by Grubbs-min -alpha .20

python run-MC-sim.py -int .1 -filter 1 -col Coeff-TPM -dist normal -folder MC-results-S4 -by Grubbs-max -alpha .01
python run-MC-sim.py -int .1 -filter 1 -col Coeff-TPM -dist normal -folder MC-results-S4 -by Grubbs-max -alpha .05
python run-MC-sim.py -int .1 -filter 1 -col Coeff-TPM -dist normal -folder MC-results-S4 -by Grubbs-max -alpha .10
python run-MC-sim.py -int .1 -filter 1 -col Coeff-TPM -dist normal -folder MC-results-S4 -by Grubbs-max -alpha .20
