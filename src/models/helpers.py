import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.externals import joblib

TITLE_FORMAT = 'Performance of a {} regressor on five second snippets'
PROJECT_ROOT = os.path.normpath(os.path.join(sys.path[0], '../../../'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

def list_absolute(directory):
    return sorted([
        os.path.join(directory, name)
        for name in os.listdir(directory)
    ])

def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, name + '.pickle'))

def save_model(model, name):
    return joblib.dump(model, os.path.join(MODEL_DIR, name + '.pickle'))

def load_data(kind, split):
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', kind, split)
    Xs = np.concatenate([
        np.load(file) for file in list_absolute(os.path.join(data_dir, 'X'))
    ])
    Ys = np.concatenate([
        np.load(file) for file in list_absolute(os.path.join(data_dir, 'Y'))
    ])

    return (Xs, Ys)

def load_data_in_chunks(kind, split, chunk_size):
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', kind, split)
    raw_xs = [
        np.load(file) for file in list_absolute(os.path.join(data_dir, 'X'))
    ]
    raw_ys = [
        np.load(file) for file in list_absolute(os.path.join(data_dir, 'Y'))
    ]
    Xs = []
    Ys = []
    for (raw_x_file, raw_y_file) in zip(raw_xs, raw_ys):
        for i_start in range(len(raw_x_file) - (chunk_size - 1)):
            Xs.append(raw_x_file[i_start:i_start + chunk_size])
            # to prevent cheating tag with the closest timestamp
            Ys.append(raw_y_file[i_start + chunk_size - 1])
    return (np.array(Xs), np.array(Ys))

def report_results(X, Y_true, Y_pred, predictor_name, outfile):
    r2 = r2_score(Y_true, Y_pred)
    (n, p) = X.shape
    r2_corrected = r2 - (1 - r2) * (p / (n - p - 1))
    print('Raw R^2: {}'.format(r2))
    print('Adjusted R^2: {}'.format(r2_corrected))

    fig, ax = plt.subplots()
    plt.scatter(Y_true, Y_pred, s=1)
    plt.title(TITLE_FORMAT.format(predictor_name))
    plt.xlabel('Actual time (seconds)')
    plt.ylabel('Predicted time (seconds)')
    ax.set_ylim(bottom=0)
    ax.set_aspect('equal')
    plt.tight_layout()
    figpath = os.path.join(PROJECT_ROOT, 'reports', 'figures', outfile)
    plt.savefig(figpath)
    plt.show()
