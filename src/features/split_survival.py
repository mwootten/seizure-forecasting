import os
import sys
import random
import numpy as np

data_dir = os.path.normpath(os.path.join(sys.path[0], '../data'))
in_dir = os.path.join(data_dir, 'interim', 'survival')
out_dir = os.path.join(data_dir, 'processed', 'survival')
basenames = os.listdir(os.path.join(data_dir, 'interim', 'survival', 'X'))
random.shuffle(basenames)
num_test = int(len(basenames) * TEST_FRACTION)
num_train = len(basenames) - num_test
labels = ['test'] * num_test + ['train'] * num_train

for (kind, file) in zip(labels, basenames):
    Xs_original = np.load(os.path.join(data_dir, 'interim', 'survival', 'X', file))
    Ys_original = np.load(os.path.join(data_dir, 'interim', 'survival', 'Y', file))
    Xs_normalized = normalize_feature_vectors(Xs_original)
    Xs_normalized.dump(os.path.join(data_dir, 'processed', 'survival', kind, 'X', file))
    Ys_original.dump(os.path.join(data_dir, 'processed', 'survival', kind, 'Y', file))
    print(file)
