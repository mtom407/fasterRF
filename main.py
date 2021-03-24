"""This module tests the faster version of random forests presented in:
https://machinelearningmastery.com/implement-random-forest-scratch-python/"""

from random import seed
import os
import numpy as np
from RF_functions import load_data, create_seq_df, run_rf

# just to be safe lmao
SEED_VALUE = 1337
os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
seed(SEED_VALUE)
np.random.seed(SEED_VALUE)

# load & transform
donors, don_targets = load_data("spliceDTrainKIS.dat")
acceptors, acc_targets = load_data("spliceATrainKIS.dat")
donorDF = create_seq_df(donors, don_targets, True)
acceptorDF = create_seq_df(acceptors, acc_targets, True)

# grid search on/off and run
grid = False
if grid:
    n_trees_vec = [1, 5]#, 10]
    depth_vec = [7, 10]#, 12]
    min_size_vec = [1, 5]#, 10]
    sample_size_vec = [0.25, 0.33]#, 0.5]

    for n_trees in n_trees_vec:
        for depth in depth_vec:
            for min_size in min_size_vec:
                for sample_size in sample_size_vec:
                    tree_count, acc = run_rf(acceptorDF, n_trees, depth, min_size, sample_size)
else:
    n_trees = 5
    depth = 7
    min_size = 5
    sample_size = 0.33

    tree_count, acc = run_rf(donorDF, n_trees, depth, min_size, sample_size)
