"""
Reads EEG files and performs the necessary segmentation and selection to pick
out appropriate segments. Calls out to `compute_features` to calculate the
actual features for segment
"""
import pyedflib
import numpy as np
from compute_features import all_features

ALLOWED_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]
SECONDS_PER_SEGMENT = 5
SAMPLE_RATE = 256
SAMPLES_PER_SEGMENT = SAMPLE_RATE * SECONDS_PER_SEGMENT
FEATURE_VECTOR_SIZE = 643

def overlaps(s1, s2):
    """
    Determines whether segments specified by their endpoints overlap.
    """
    (i1, f1) = s1
    (i2, f2) = s2
    return ((i1 < f2) and (i2 < f1)) or ((i2 < f1) and (i1 < f2))

def any_overlap(t_i, t_f, segments):
    """
    Extends `overlaps` to check against an artitrary number of segments at once
    """
    return any(overlaps((t_i, t_f), seg) for seg in segments)

def select_segments(num_segments, seizure_times):
    """
    Selects segments that meet the criteria for inclusion in the model:
    * They don't overlap with a seizure (we care about preictal activity only)
    * It is possible to determine the duration until the next seizure. That
      means that the segment must be discarded if it is after the last seizure
      in any given file, because no one knows whether the next seizure was
      15 minutes or 3 hours after the end.

     Also, split up the recording based on which seizures it precedes. This is
     to make sure that within each stretch of segments, segments which are
     consecutive in the array are consecutive in real life. This would prevent,
     for example, a snippet one second before a seizure from appearing next to
     the postictal segment 35 minutes before the next seizure. While this does
     not matter if the model inputs are single segments, this would screw up
     convolutional or other sequential models.

    Outputs: an array of pairs of the following two variables, one for each
    seizure in the given file:

    * `indices`: the indices of the segments in the original file
    * `Y`: the time until the next seizure for each segment; corresponds to
      positions in the `indices` array

    """
    num_seizures = len(seizure_times)
    indices = [[] for _ in range(num_seizures)]
    Y = [[] for _ in range(num_seizures)]
    for segment_num in range(num_segments):
        t_i = SECONDS_PER_SEGMENT * (segment_num + 0)
        t_f = SECONDS_PER_SEGMENT * (segment_num + 1)
        overlaps_seizure = any_overlap(t_i, t_f, seizure_times)
        time_before_each_seizure = [
            # the time from the end of the segment to the seizure start
            (st_i - t_f, seizure_index)
            # for all of the seizures
            for (seizure_index, (st_i, st_f)) in enumerate(seizure_times)
            # if the segment ends before the seizure begins
            if t_f <= st_i
        ]
        # If no segments meet the requirements, then just add None
        # otherwise,
        (time_before_last_seizure, last_seizure_index) = \
            min(time_before_each_seizure, default=(None, None))
        if (not overlaps_seizure) and time_before_last_seizure is not None:
            indices[last_seizure_index].append(segment_num)
            Y[last_seizure_index].append(time_before_last_seizure)
    return list(zip(indices, Y))

def read_file(name, seizure_times):
    """
    Reads in an EEG in EDF format stored at `name` and applies the block
    selection rules described in `select_segments`. The output is an array of
    (X, Y) pairs containing the inputs and outputs needed for training a machine
    learning model on these values (can be directly input into a learner with
    the `scikit-learn` API). Each pair corresponds to a single seizure within
    the file.
    """
    f = pyedflib.EdfReader(name)
    labels = f.getSignalLabels()
    channel_indices = [labels.index(x) for x in ALLOWED_CHANNELS]
    # Make sure that every recording has the same length
    assert len(set(f.getNSamples())) == 1
    sample_count = f.getNSamples()[0]
    num_segments = int(sample_count / SAMPLES_PER_SEGMENT)
    input_output_pairs = []
    for (segment_indices, outputs) in select_segments(num_segments, seizure_times):
        file_segments = np.empty((
            len(channel_indices), # indexed first by channel
            len(segment_indices),  # then by segment
            SAMPLES_PER_SEGMENT # and last by time stamp within a segment
        ))
        for (dst_index, src_index) in enumerate(channel_indices):
            raw = f.readSignal(src_index)
            # Chop off everything after the last full five second segment
            # This chunk will never be valid, because there cannot possibly be
            # a seizure after it
            total_samples = num_segments * SAMPLES_PER_SEGMENT
            raw = raw[:total_samples]
            segmented_reading = raw.reshape((num_segments, SAMPLES_PER_SEGMENT))
            file_segments[dst_index] = segmented_reading[segment_indices]
        # transpose to segment, channel, subparts
        file_segments_transposed = np.transpose(file_segments, (1, 0, 2))
        inputs = np.empty((len(segment_indices), FEATURE_VECTOR_SIZE))
        for (segment_index, segment) in enumerate(file_segments_transposed):
            inputs[segment_index] = all_features(segment)
        input_output_pairs.append((inputs, outputs))
    return input_output_pairs
