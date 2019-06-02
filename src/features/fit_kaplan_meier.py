import os
import sys
import numpy as np
from select_common import select_segments_broad, read_annotations
from lifelines import KaplanMeierFitter
import pickle

data_dir = os.path.normpath(os.path.join(sys.path[0], '../data-new/'))
out_file = os.path.join(data_dir, 'interim', 'survival-model.pickle')

annotations = read_annotations(data_dir)
durations = []
observeds = []

for annotation in annotations.values():
    (indices, Y) = select_segments_broad(annotation)
    for section in Y[:-1]:
        durations += section
        observeds += [True] * len(section)
    durations += Y[-1]
    observeds += [False] * len(Y[-1])

km = KaplanMeierFitter()
km = km.fit(durations, event_observed=observeds)
with open(out_file, 'wb') as out_handle:
    pickle.dump(km, out_handle)
