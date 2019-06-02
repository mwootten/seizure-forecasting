from sklearn.ensemble import RandomForestRegressor

import sys
sys.path.append('..')
from helpers import load_data, save_model

(Xs, Ys) = load_data('basic', 'train')
regr = RandomForestRegressor(n_estimators=100)
regr.fit(Xs, Ys)
save_model(regr, 'random-forest')
