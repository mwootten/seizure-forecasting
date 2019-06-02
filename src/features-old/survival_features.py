from lifelines import KaplanMeierFitter
from read_annotations import parse_summary_file
import json
import os
import sys
import numpy as np

SECONDS_PER_SEGMENT = 5
# Deciles of the training Ys
CUTOFFS = np.array(
    [243, 514, 822, 1162, 1557, 2038, 2725, 3981, 6102, 13683, np.inf]
)


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

def select_segments_part(num_segments, seizure_times):
    end_time = num_segments * SECONDS_PER_SEGMENT
    num_seizures = len(seizure_times)
    indices = [[] for _ in range(num_seizures + 1)]
    Y = [[] for _ in range(num_seizures + 1)]
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
        if overlaps_seizure:
            pass
        elif time_before_last_seizure is None:
            # Censorship!
            indices[num_seizures].append(segment_num)
            Y[-1].append(end_time - t_f)
        else:
            indices[last_seizure_index].append(segment_num)
            Y[last_seizure_index].append(time_before_last_seizure)

    return (indices, Y)

def convert_survival_form(lengths, annotations):
    duration = []
    observed = []
    for (length, annotation) in zip(lengths, annotations):
        (indices, Y) = select_segments_part(length, annotation)
        assert len(Y) > 0
        for seizure in Y[:-1]:
            duration.append(seizure)
            observed.append([True] * len(seizure))
        duration.append(Y[-1])
        observed.append([False] * len(Y[-1]))
    return (duration, observed)

def build_kaplan_meier(lengths, annotations):
    duration = []
    observed = []
    for (length, annotation) in zip(lengths, annotations):
        (indices, Y) = select_segments_part(length, annotation)
        assert len(Y) > 0
        for seizure in Y[:-1]:
            duration += seizure
            observed += [True] * len(seizure)
        duration += Y[-1]
        observed += [False] * len(Y[-1])
    kmf = KaplanMeierFitter()
    kmf.fit(duration, event_observed=observed)
    return kmf

def encode_target(duration, observed, kmf):
    """
    encoding = (duration > CUTOFFS).astype(int) * 2 - 1
    if not observed:
        kmf.predict()
    """
    return np.zeros(11)


def get_targets(lengths, annotations, kmf):
    targets = []
    survival_form = zip(*convert_survival_form(lengths, annotations))
    for (durations, observeds) in survival_form: # for each seizure
        seizure_targets = []
        for (duration, observed) in zip(durations, observeds): # for each snippet
            target = encode_target(duration, observed, kmf)
            seizure_targets.append(target)
        targets.append(seizure_targets)
    return targets

def select_segments(num_segments, seizure_times):
    (indices, Y) = select_segments_part(num_segments, seizure_times)
    

def read_file(name, seizure_times):
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



# ---------------------------------------------------------------------------- #

def list_absolute(directory):
    return sorted([
        os.path.join(directory, name)
        for name in os.listdir(directory)
    ])

data_dir = os.path.normpath(os.path.join(sys.path[0], '../../data/'))
summary_files = list_absolute(os.path.join(data_dir, 'external', 'summaries'))
summaries = dict()
for summary_file in summary_files:
    summaries.update(parse_summary_file(summary_file))
summaries = dict(sorted(summaries.items()))

with open('lengths.json', 'r') as f:
    lengths = json.load(f)
lengths = dict(sorted((k, v) for (k, v) in lengths.items() if k in summaries))

kmf = build_kaplan_meier(lengths.values(), summaries.values())
kmf.plot()
