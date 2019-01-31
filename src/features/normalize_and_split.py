import os
import sys
import random
import numpy as np

TEST_FRACTION = 0.2

data_dir = os.path.normpath(os.path.join(sys.path[0], '../../data/'))
basenames = os.listdir(os.path.join(data_dir, 'interim', 'X'))
random.shuffle(basenames)
all_features = np.concatenate([
    np.load(os.path.join(data_dir, 'interim', 'X', basename))
    for basename in basenames
])
means = all_features.mean(axis=0)
stdevs = all_features.std(axis=0)

num_test = int(len(basenames) * TEST_FRACTION)
num_train = len(basenames) - num_test
labels = ['test'] * num_test + ['train'] * num_train

for (kind, file) in zip(labels, basenames):
    Xs_original = np.load(os.path.join(data_dir, 'interim', 'X', file))
    Ys_original = np.load(os.path.join(data_dir, 'interim', 'Y', file))
    Xs_normalized = (Xs_original - means) / stdevs
    Xs_normalized.dump(os.path.join(data_dir, 'processed', kind, 'X', file))
    Ys_original.dump(os.path.join(data_dir, 'processed', kind, 'Y', file))
