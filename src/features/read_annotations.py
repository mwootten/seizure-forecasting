"""
Read the (seperate) annotations marking the locations of the seizures within the
EDF files.

This module currently does not use the binary .seizures files because we could
not find any documentation on the format. However, it appears that the same
information is encoded in a slightly less convenient form in the text summaries.

"""

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
                t_i = int(endpoints_text[i+0].split(" ")[-2])
                t_f = int(endpoints_text[i+1].split(" ")[-2])
                endpoints.append((t_i, t_f))
            records[file_name] = endpoints
        return records
