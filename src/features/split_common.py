import os
import sys
import random
import numpy as np
import scipy.stats
from statsmodels.distributions.empirical_distribution import ECDF

data_dir = os.path.normpath(os.path.join(sys.path[0], '../data'))
in_dir = os.path.join(data_dir, 'interim', 'survival')

# The 703239 below is the total number of snippets in the whole dataset
# This obviously is not stable across datasets
# The 624, however, should be, since it's the size of a feature vector.
all_features = np.zeros((703239, 624), dtype='float64')
index = 0

for file in os.listdir(os.path.join(in_dir, 'X')):
    xs_part = np.load(os.path.join(in_dir, 'X', file))
    all_features[index:index+len(xs_part)] = xs_part
    index += len(xs_part)

# Should be a no-op, but just to make sure I'm not forgetting anything...
all_features = all_features[:index]
cdfs = [ECDF(values) for values in all_features.T]
del all_features
return cdfs

def normalize_feature_vectors(Xs_original):
    Xs_transformed = np.full_like(Xs_original, np.nan)
    for (i, (x, ecdf)) in enumerate(zip(Xs_original.T, cdfs)):
        # The nudging is to prevent the highest/lowest value from getting a
        # quantile of exactly +/- 1, which would have a z-score of +/- infinity
        x_nudged = x + 0.000000001 * np.ptp(x) * np.sign(np.median(x) - x)
        Xs_transformed[:, i] = scipy.stats.norm.ppf(ecdf(x_nudged))
    return Xs_transformed

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
