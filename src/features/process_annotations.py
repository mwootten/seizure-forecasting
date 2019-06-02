"""
Read the (seperate) annotations marking the locations of the seizures within the
EDF files, as well as the durations.

This module currently does not use the binary .seizures files because we could
not find any documentation on the format. However, it appears that the same
information is encoded in a slightly less convenient form in the text summaries.
Due to the difficulty of parsing out the seizure start and end times, I have
elected to just read the EDF files themselves to get the total length.
"""

import os
import sys
import json
import pyedflib

def list_absolute(directory):
    return sorted([
        os.path.join(directory, name)
        for name in os.listdir(directory)
    ])

def parse_summary_file(name):
    """
    Parse the chb**-summary.txt files from the CHB-MIT dataset.
    """
    with open(name, 'r') as file:
        text = file.read().strip()
        body = text[text.index("File Name"):]
        raw_records = body.split("\n\n")
        records = dict()
        for raw_record in raw_records:
            if 'Channels changed' in raw_record:
                continue
            parts = raw_record.strip().split('\n')
            file_name = parts[0].split(': ')[1]
            endpoints_text = parts[4:]
            endpoints = []
            for i in range(len(endpoints_text) // 2):
                t_i = int(endpoints_text[2*i+0].split(" ")[-2])
                t_f = int(endpoints_text[2*i+1].split(" ")[-2])
                endpoints.append((t_i, t_f))
            records[file_name] = endpoints
        return records

data_dir = os.path.normpath(os.path.join(sys.path[0], '../data/'))
eeg_dir = os.path.join(data_dir, 'external', 'recordings')
summary_dir = os.path.join(data_dir, 'external', 'summaries')

summaries = dict()
for summary_file in list_absolute(summary_dir):
    summaries.update(parse_summary_file(summary_file))

annotations = dict()
for eeg_basename in summaries.keys():
    f = pyedflib.EdfReader(os.path.join(eeg_dir, eeg_basename))
    assert len(set(f.getNSamples())) == 1
    assert len(set(f.getSampleFrequencies())) == 1
    sample_count = f.getNSamples()[0]
    sample_rate = f.getSampleFrequency(0)
    annotations[eeg_basename] = {
        'length': int(sample_count / sample_rate),
        'seizure_times': summaries.get(eeg_basename, default=[])
    }

print(json.dumps(annotations))
