import sys
sys.path.append('..')
from helpers import load_data_in_chunks, load_model, report_results

(Xs, Ys) = load_data_in_chunks('test', chunk_size=5)
Xs = Xs.reshape(Xs.shape[0], -1)
regr = load_model('linear-multiple')
Ys_pred = regr.predict(Xs)
report_results(Xs, Ys, Ys_pred, 'linear', 'linear-multiple.svg')
