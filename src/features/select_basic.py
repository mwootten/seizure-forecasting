import os
import sys
import numpy as np
from select_common import select_segments_broad, read_annotations

data_dir = os.path.normpath(os.path.join(sys.path[0], '../../data/'))
in_dir = os.path.join(data_dir, 'interim', 'complete')
out_dir = os.path.join(data_dir, 'interim', 'basic')

annotations = read_annotations(data_dir)

for basename in annotations.keys():
    Xs_all = np.load(os.path.join(in_dir, 'X', basename))
    (indices, Ys) = select_segments_broad(annotations[basename])
    indices = indices[:-1]
    Ys = Ys[:-1]
    for (i, (seizure_indices, seizure_Ys)) in enumerate(zip(indices, Ys)):
        X = Xs_all[seizure_indices]
        Y = seizure_Ys
        (root, ext) = os.path.splitext(basename)
        new_basename = root + '_' + str(i) + '.npy'
        np.array(X).dump(os.path.join(out_dir, 'X', new_basename))
        np.array(Y).dump(os.path.join(out_dir, 'Y', new_basename))
    print('SUCCESS: ' + basename)
