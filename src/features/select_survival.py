import os
import sys
import pickle
import numpy as np
from select_common import select_segments_broad, read_annotations

data_dir = os.path.normpath(os.path.join(sys.path[0], '../data'))
in_dir = os.path.join(data_dir, 'interim', 'complete')
out_dir = os.path.join(data_dir, 'interim', 'survival')
km_file = os.path.join(data_dir, 'interim', 'survival-model.pickle')
annotations = read_annotations(data_dir)
with open(km_file, 'rb') as f:
    km = pickle.load(f)
CUTOFFS = np.array(
    [243, 514, 822, 1162, 1557, 2038, 2725, 3981, 6102, 13683, np.inf]
)
EXCLUDE_FILES = ['chb12_27.edf', 'chb12_28.edf', 'chb12_29.edf']


def encode_target(duration, observed):
    encoding = (duration > CUTOFFS).astype(float) * 2 - 1
    if not observed:
        rewrite_flags = (duration <= CUTOFFS)
        new_values = 2 * km.predict(CUTOFFS[rewrite_flags] - duration) - 1
        encoding[rewrite_flags] = new_values
    return encoding

for basename in annotations.keys():
    if basename in EXCLUDE_FILES:
        continue
    Xs_all = np.load(os.path.join(in_dir, 'X', basename.replace("edf", "npy")))
    (indices, Ys) = select_segments_broad(annotations[basename])
    observed_flag = [True] * (len(Ys) - 1) + [False]
    specs = enumerate(zip(indices, Ys, observed_flag))
    for (i, (section_indices, section_durations, section_observed)) in specs:
        (root, ext) = os.path.splitext(basename)
        new_basename = root + '_' + str(i) + '.npy'
        Xpath = os.path.join(out_dir, 'X', new_basename)
        Ypath = os.path.join(out_dir, 'Y', new_basename)
        if os.path.exists(Xpath):
            continue
        X = Xs_all[section_indices]
        Y = np.array([
            encode_target(duration, section_observed)
            for duration in section_durations
        ])
        np.array(X).dump(Xpath)
        np.array(Y).dump(Ypath)
    print('SUCCESS: ' + basename)
