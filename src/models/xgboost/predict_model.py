import sys
sys.path.append('..')
from helpers import load_data, load_model, report_results

(Xs, Ys) = load_data('test')
regr = load_model('xgb')
Ys_pred = regr.predict(Xs)
report_results(Xs, Ys, Ys_pred, 'gradient boosted', 'xgboost.svg')
