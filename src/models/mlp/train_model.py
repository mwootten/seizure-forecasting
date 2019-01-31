from sklearn.neural_network import MLPRegressor
import numpy as np

import sys
sys.path.append('..')
from helpers import load_data, save_model

(Xs, Ys) = load_data('train')
regr = MLPRegressor(hidden_layer_sizes=(60, 10), verbose=True, max_iter=1000)
regr.fit(Xs, Ys)
save_model(regr, 'mlp')
