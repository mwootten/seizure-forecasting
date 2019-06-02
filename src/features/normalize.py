import os
import numpy as np
import scipy.stats
from statsmodels.distributions.empirical_distribution import ECDF


# test: 142243
# train: 560996

all_features = np.zeros((560996, 624), dtype='float64')
index = 0

for file in os.listdir():
    xs_part = np.load(file)
    all_features[index:index+len(xs_part)] = xs_part
    index += len(xs_part)

cdfs = [ECDF(values) for values in all_features.T]
del all_features

def normalize_feature_vectors(Xs_original):
    Xs_transformed = np.full_like(Xs_original, np.nan)
    for (i, (x, ecdf)) in enumerate(zip(Xs_original.T, cdfs)):
        # The nudging is to prevent the highest/lowest value from getting a
        # quantile of exactly +/- 1, which would have a z-score of +/- infinity
        x_nudged = x + 0.000000001 * np.ptp(x) * np.sign(np.median(x) - x)
        Xs_transformed[:, i] = scipy.stats.norm.ppf(ecdf(x_nudged))
    return Xs_transformed


for file in os.listdir():
    Xs_original = np.load(file)
    Xs_normalized = normalize_feature_vectors(Xs_original)
    Xs_normalized.dump(file + '-normalized')
