import os
import sys
import numpy as np
from read_eeg import read_file
from read_annotations import parse_summary_file

def list_absolute(directory):
    return sorted([
        os.path.join(directory, name)
        for name in os.listdir(directory)
    ])

data_dir = os.path.normpath(os.path.join(sys.path[0], '../../data/'))

# Read in the summaries
summary_files = list_absolute(os.path.join(data_dir, 'external', 'summaries'))
summaries = dict()
for summary_file in summary_files:
    summaries.update(parse_summary_file(summary_file))

eeg_files = list_absolute(os.path.join(data_dir, 'external', 'recordings'))
for eeg_file in eeg_files:
    try:
        basename = os.path.basename(eeg_file)
        seizure_times = summaries[basename]
        input_output_pairs = read_file(eeg_file, seizure_times)
        for (seizure_index, (X, Y)) in enumerate(input_output_pairs):
            (root, ext) = os.path.splitext(basename)
            new_basename = root + '_' + str(seizure_index) + '.npy'
            X_file = os.path.join(data_dir, 'interim', 'X', new_basename)
            Y_file = os.path.join(data_dir, 'interim', 'Y', new_basename)
            np.array(X).dump(X_file)
            np.array(Y).dump(Y_file)
        print('SUCCESS: ' + basename)
    except:
        print('FAILURE: ' + os.path.basename(eeg_file))
