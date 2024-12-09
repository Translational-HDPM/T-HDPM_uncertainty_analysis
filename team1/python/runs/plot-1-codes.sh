#!/bin/bash

python run-MC-sim.py -int .1 -filter 0 -dist LHS -folder Team-3-Scores
python run-MC-sim.py -int .1 -filter 100 -dist LHS -folder Team-3-Scores
python run-MC-sim.py -int .1 -filter 200 -dist LHS -folder Team-3-Scores
python run-MC-sim.py -int .1 -filter 400 -dist LHS -folder Team-3-Scores
python run-MC-sim.py -int .1 -filter 800 -dist LHS -folder Team-3-Scores
