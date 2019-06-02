import os
import sys
import tqdm
import random
import numpy as np
import scipy.stats
from statsmodels.distributions.empirical_distribution import ECDF

TEST_FRACTION = 0.2

def js_distance(P, Q):
    overall_min = min(P.min(), Q.min())
    overall_max = max(P.max(), Q.max())
    kP = scipy.stats.gaussian_kde(P)
    kQ = scipy.stats.gaussian_kde(Q)
    xs = np.linspace(overall_min, overall_max, 100)
    Psmooth = kP(xs)
    Qsmooth = kQ(xs)
    Msmooth = (Psmooth + Qsmooth) / 2
    KL_PM = scipy.stats.entropy(Psmooth, Msmooth)
    KL_QM = scipy.stats.entropy(Qsmooth, Msmooth)
    return 0.5 * (KL_PM + KL_QM)

data_dir = os.path.normpath(os.path.join(sys.path[0], '../../data/'))
basenames = os.listdir(os.path.join(data_dir, 'interim', 'X'))
num_test = int(len(basenames) * TEST_FRACTION)
num_train = len(basenames) - num_test
labels = ['test'] * num_test + ['train'] * num_train

# Pick a train-test split such that the training and testing distributions look
# as similar as possible. With a much larger amount of data, this would be
# unnecessary; however, there are only a fairly small number of seizures, and so
# we can't necessarily rely on random sampling alone.

distances = []
for i in tqdm.trange(200):
    random.shuffle(basenames)
    Ys_clustered = {'train': [], 'test': []}
    for (kind, file) in zip(labels, basenames):
        Ys_original = np.load(os.path.join(data_dir, 'interim', 'Y', file))
        Ys_clustered[kind] += list(Ys_original)
    train_dist = np.array(Ys_clustered['train'])
    test_dist = np.array(Ys_clustered['test'])
    distances.append((js_distance(train_dist, test_dist), basenames))

basenames = min(distances)[1]

all_features = np.concatenate([
    np.load(os.path.join(data_dir, 'interim', 'X', basename))
    for basename in basenames
])

cdfs = [ECDF(values) for values in all_features.T]

def normalize_feature_vectors(Xs_original):
    Xs_transformed = np.full_like(Xs_original, np.nan)
    for (i, (x, ecdf)) in enumerate(zip(Xs_original.T, cdfs)):
        # The nudging is to prevent the highest/lowest value from getting a
        # quantile of exactly +/- 1, which would have a z-score of +/- infinity
        x_nudged = x + 0.000000001 * np.ptp(x) * np.sign(np.median(x) - x)
        Xs_transformed[:, i] = scipy.stats.norm.ppf(ecdf(x_nudged))
    return Xs_transformed

for (kind, file) in zip(labels, basenames):
    Xs_original = np.load(os.path.join(data_dir, 'interim', 'X', file))
    Ys_original = np.load(os.path.join(data_dir, 'interim', 'Y', file))
    Xs_normalized = normalize_feature_vectors(Xs_original)
    Xs_normalized = np.concatenate([Xs_normalized[:, :585], Xs_normalized[:, 603:639], Xs_normalized[:, 640:]], axis=1)
    Xs_normalized.dump(os.path.join(data_dir, 'processed', kind, 'X', file))
    Ys_original.dump(os.path.join(data_dir, 'processed', kind, 'Y', file))
    print(file)
