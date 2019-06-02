import sys
sys.path.append('..')
from helpers import load_data, load_model, report_results

import numpy as np

(Xs, Ys) = load_data('basic', 'test')
regr = load_model('mlp')
Ys_pred = regr.predict(Xs)
report_results(Xs, Ys, Ys_pred, 'multi-layer perceptron', 'mlp.svg')
