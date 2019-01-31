from sklearn.dummy import DummyRegressor

import sys
sys.path.append('..')
from helpers import load_data, save_model

(Xs, Ys) = load_data('train')
regr = DummyRegressor(strategy='median')
regr.fit(Xs, Ys)
save_model(regr, 'dummy')
