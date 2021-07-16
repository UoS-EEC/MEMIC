#!/usr/bin/env python3

import math
import os
import re
import shutil
import sys
from pathlib import Path
from dataclasses import dataclass, field
from pprint import pprint

import numpy as np
import pandas as pd

USAGE = """

workload-completion-times.py

A script to parse CompletionTimes.csv and produce a CSV that summarizes the
results for use in a PGFPlots table in the MEMIC paper.

USAGE:

    ./workload-completion-times.py <input_csv> <output_directory>

"""


#@dataclass(Frozen=False, Order=True)
@dataclass
class WorkLoadEntry:
    name: str
    start_row: int
    end_row: int
    data: pd.DataFrame = field(default_factory=pd.DataFrame)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(USAGE)
        exit(1)

    input_file = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    with open(input_file, 'r') as infile:
        input_data = infile.read()

    # Find workload entries
    workloads = []
    for lineno, line in enumerate(input_data.split('\n')):
        #if len(line.split(',')) < 3 or line == '':
        if ',,' in line or line == '' or ',' not in line:  # New workload, or last line
            if lineno == 0:
                name = line.split(',')[0]
                startrow = 1
            else:
                # Save previous workload
                workloads.append(WorkLoadEntry(name, startrow, lineno - 1))
            name = line.split(',')[0]
            startrow = lineno + 1
            name = line.split(',')[0]

    # Load one data frame per workload
    print(f"Found {len(workloads)} workloads:")
    for w in workloads:
        w.data = pd.read_csv(input_file,
                             skiprows=w.start_row,
                             nrows=w.end_row - w.start_row,
                             skipinitialspace=True).fillna(np.inf)
        print(f"\t{w.name}: {w.start_row}->{w.end_row}")

    # ------ Assemble output data ------
    #          |           |           | method1   | method2    | method3 ...
    # workload | code_size | data_size | c1 | c2 ..| c1 | c2 ..

    icachestring = '4kB (16|2|128)'
    dcachestring = '2kB (32|2|32)'

    methods = [f'{icachestring}-{v}' for v in ['AllocatedState', 'Freezer']] \
            + [f"{icachestring}-{dcachestring}"]

    columns = ['Workload', 'code (kB)', 'data (kB)']

    capacitances = [1.0e-6, 4.7e-6, 10e-6]

    results = []

    for w in workloads:
        # one row
        name = w.name
        csize = w.data['code size (kB)'][0]
        dsize = w.data['data size (kB)'][0]
        row = [name, str(csize), str(dsize)]

        for m in methods:
            p = w.data[w.data['icache config(LW|NL|NS)'] == m]
            print(p)
            for c in capacitances:
                v = p[p['capacitance (F)'] == c]['mean completion-time (s)']
                print(f"{w.name} {m} -- {c}")
                print(v)
                print(float(v))
                row.append(f"{float(v):.2f}")

        results.append(row)

    # Emit csv
    methodnames = [r'I-AS', r'I-FZ', 'MEMIC']
    with open(output_path / 'completion_times.csv', 'w+') as of:
        #of.write(',,,')
        #of.write(','.join(methods) + '\n')
        #of.write(','.join(columns) + ',')
        #of.write(','.join('mean,stdev' for _ in methods) + '\n')

        of.write(','.join(' ' for _ in columns))
        of.write(', ' + ', , ,'.join(methodnames) + ', , ' + '\n')
        of.write(','.join(columns) + ',')
        caps = r'\SI{4.7}{\micro\farad}, \SI{10}{\micro\farad}, \SI{100}{\micro\farad}'
        caps = r'1.0 uF ,4.7 uF, 10 uF'
        of.write(','.join(caps for _ in methodnames) + '\n')

        for r in results:
            of.write(','.join(r) + '\n')
            print(','.join(r))
    exit(0)
