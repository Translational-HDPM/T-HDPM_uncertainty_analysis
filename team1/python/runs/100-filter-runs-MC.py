#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 16:10:20 2023

@author: svaddadi
"""
import os
import multiprocessing as mp

def runFcn(i):
    os.system('python ../py-files/run-MC-sim.py -run {} --filter 100 -int .1 -folder MC-results-S3'.format(i))


with mp.Pool() as pool:
    pool.map(runFcn, list(range(1000)))



