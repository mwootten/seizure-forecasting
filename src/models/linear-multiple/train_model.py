from sklearn.linear_model import LinearRegression

import sys
sys.path.append('..')
from helpers import load_data_in_chunks, save_model

(Xs, Ys) = load_data_in_chunks('train', chunk_size=5)
regr = LinearRegression()
regr.fit(Xs.reshape(Xs.shape[0], -1), Ys)
save_model(regr, 'linear-multiple')
