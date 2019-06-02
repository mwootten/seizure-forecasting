from sklearn.linear_model import LinearRegression

import sys
sys.path.append('..')
from helpers import load_data, save_model

(Xs, Ys) = load_data('basic', 'train')
regr = LinearRegression()
regr.fit(Xs, Ys)
save_model(regr, 'linear-basic')
