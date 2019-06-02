import os
import sys
import pyedflib
import numpy as np
from read_eeg import read_file
from joblib import Parallel, delayed
from compute_features import all_features
from read_annotations import parse_summary_file


ALLOWED_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]
SECONDS_PER_SEGMENT = 5
SAMPLE_RATE = 256
SAMPLES_PER_SEGMENT = SAMPLE_RATE * SECONDS_PER_SEGMENT
FEATURE_VECTOR_SIZE = 624

def read_file(name):
    f = pyedflib.EdfReader(name)
    labels = f.getSignalLabels()
    channel_indices = [labels.index(x) for x in ALLOWED_CHANNELS]
    # Make sure that every recording has the same length
    assert len(set(f.getNSamples())) == 1
    sample_count = f.getNSamples()[0]
    num_segments = int(sample_count / SAMPLES_PER_SEGMENT)
    total_samples = num_segments * SAMPLES_PER_SEGMENT

    file_segments = np.empty((
        len(channel_indices), # indexed first by channel
        num_segments,  # then by segment
        SAMPLES_PER_SEGMENT # and last by time stamp within a segment
    ))
    for (dst_index, src_index) in enumerate(channel_indices):
        raw = f.readSignal(src_index)
        raw = raw[:total_samples]
        segmented_reading = raw.reshape((num_segments, SAMPLES_PER_SEGMENT))
        file_segments[dst_index] = segmented_reading
    # transpose to segment, channel, subparts
    file_segments_transposed = np.transpose(file_segments, (1, 0, 2))
    inputs = np.array(Parallel(n_jobs=4)(delayed(all_features)(segment) for segment in file_segments_transposed))
    return inputs

# ---------------------------------------------------------------------------- #

def list_absolute(directory):
    return sorted([
        os.path.join(directory, name)
        for name in os.listdir(directory)
    ])

data_dir = os.path.normpath(os.path.join(sys.path[0], '../../data/'))
in_dir = os.path.join(data_dir, 'interim', 'complete', 'X')
out_dir = os.path.join(data_dir, 'external', 'recordings')

eeg_files = list_absolute('/run/media/mwootten/Data/CHB')
for eeg_file in eeg_files:
    basename = os.path.basename(eeg_file)
    try:
        (root, ext) = os.path.splitext(basename)
        X_file = os.path.join(in_dir, root + '.npy')
        if os.path.exists(X_file):
            continue
        X = read_file(eeg_file)
        np.array(X).dump(X_file)
        print('SUCCESS: ' + basename)
    except:
        print('FAILURE: ' + basename)
