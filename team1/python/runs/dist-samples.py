import os

for dists in ['Ball','uniform-hypercube','grid-hypercube','cvt-hypercube','multimodal-hypercube','cross-shape','curve-sample','stripes-sample']:
    os.system('python run-MC-sim.py -dist {} -filter 0 -csv ../data/AD_sort_by_AD_over_NCI_v3_pop.csv'.format(dists))
