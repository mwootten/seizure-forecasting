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

def select_segments_part(num_segments, seizure_times): # -> (indices, Y)
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

def expand_Ys(Ys):
    expanded_Ys = [[] for _ in range(len(Ys))]
    for original_Ys in Ys[:-1]:
        expanded_Ys.append([encode_target(Y, True) for Y in original_Ys])
    expanded_Ys.append([encode_target(Y, False) for Y in Ys[-1]])
    return expanded_Ys

def encode_target(duration, observed):
    return [0] * 11
