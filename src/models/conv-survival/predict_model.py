import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import sys
sys.path.append('..')
from helpers import load_data_in_chunks, load_model, report_results
from helpers import PROJECT_ROOT
FIGURE_PATH = os.path.join(PROJECT_ROOT, 'reports', 'figures')

(Xs, Ys) = load_data_in_chunks('survival', 'test', chunk_size=5)
Xs = Xs.astype('float32')[:100]
Ys = Ys.astype('float32')[:100]
regr = load_model('conv-survival')
Ys_pred = regr.predict(Xs)

Ys_buckets = [y.nonzero()[0].min(initial=11) for y in (Ys < 0.5)]
Ys_pred_buckets = [y.nonzero()[0].min(initial=11) for y in (Ys_pred < 1)]

pairs = Counter(zip(Ys_buckets, Ys_pred_buckets))
map = np.zeros((12, 12), dtype=int)
for ((actual, predicted), count) in pairs.items():
    map[actual, predicted] = count
sns.heatmap(map)
plt.xlabel('Predicted bucket')
plt.ylabel('Actual bucket')
plt.savefig(os.path.join(FIGURE_PATH, 'conv-survival.svg'))
