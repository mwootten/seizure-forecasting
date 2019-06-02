from xgboost import XGBRegressor

import sys
sys.path.append('..')
from helpers import load_data, save_model

(Xs, Ys) = load_data('basic', 'train')
regr = XGBRegressor(silent=False)
regr.fit(Xs, Ys)
save_model(regr, 'xgb')
