import sys
sys.path.append('..')
from helpers import load_data, load_model, report_results

(Xs, Ys) = load_data('test')
regr = load_model('random-forest')
Ys_pred = regr.predict(Xs)
report_results(Xs, Ys, Ys_pred, 'random forest', 'rf.svg')
