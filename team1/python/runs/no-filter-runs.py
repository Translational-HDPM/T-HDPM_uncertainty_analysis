#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 16:10:20 2023

@author: svaddadi
"""
import os

for i in range(50):
    os.system('python ../py-files/run-MC-sim.py -run {} --filter 0 -int .1'.format(i))

for i in range(50):
    os.system('python ../py-files/run-MC-sim.py -run {} --filter 0 -int .2'.format(i))
    

