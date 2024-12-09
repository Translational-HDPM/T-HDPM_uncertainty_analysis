#!/bin/bash


python run-MC-sim.py -int .1 -filter 100 -col Avg-TPM -dist normal -folder MC-results-S4 -by largest
python run-MC-sim.py -int .1 -filter 200 -col Avg-TPM -dist normal -folder MC-results-S4 -by largest
python run-MC-sim.py -int .1 -filter 400 -col Avg-TPM -dist normal -folder MC-results-S4 -by largest
python run-MC-sim.py -int .1 -filter 800 -col Avg-TPM -dist normal -folder MC-results-S4 -by largest

python run-MC-sim.py -int .1 -filter 100 -dist normal -folder MC-results-S4 -by largest
python run-MC-sim.py -int .1 -filter 200 -dist normal -folder MC-results-S4 -by largest
python run-MC-sim.py -int .1 -filter 400 -dist normal -folder MC-results-S4 -by largest
python run-MC-sim.py -int .1 -filter 800 -dist normal -folder MC-results-S4 -by largest