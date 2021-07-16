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

cache-epi.py

A script to parse [Data|Instruction]CacheStudy.csv and produce csv's suitable
for TikZ/PGFPlots plotting in the MEMIC paper.

USAGE:

    ./cache-epi.py <input_csv> <output_directory>

"""


#@dataclass(Frozen=False, Order=True)
@dataclass
class WorkLoadEntry:
    name: str
    start_row: int
    end_row: int
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def best_config(self):
        """
        Find and return config that has the smallest ontime+chargetime
        """
        df = self.data.where(self.data['vwarn'] == 2.1
                             ) if 'vwarn' in self.data.columns else self.data
        idx = (df['on-time (s)'] + df['charge-time (s)']).idxmin()
        return df.iloc[idx, 0]


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
        print(
            f"\t{w.name}: {w.start_row}->{w.end_row}: best config = {w.best_config}"
        )

    # Find the best overall configuration, based on geometric mean of the total completion time
    getConfigColumn = lambda df: df[df.columns[0]]
    from scipy.stats.mstats import gmean
    bestScore = float('Inf')
    bestScoreIdx = 0
    bestScoreConfig = ''
    for i, config in enumerate(getConfigColumn(workloads[0].data)):
        score = gmean([
            w.data['on-time (s)'][i] + w.data['charge-time (s)'][i]
            for w in workloads
            # Some workloads will fail here because we're using AS
            # so just count the workloads that actually succeed
            if not np.isinf(w.data['on-time (s)'][i])
        ])
        if score < bestScore:
            bestScore = score
            bestScoreIdx = i
            bestScoreConfig = config
    print(f"Overall best config {bestScoreConfig} scores {bestScore}")

    norm_completion_times = {}
    # ------ Emit output csvs ------
    if 'Data' in input_file.name:
        # Energies & runtimes
        norm_completion_times = {
            'AllocatedState': [],
            'Freezer': [],
            bestScoreConfig: []
        }

        for w in workloads:
            df = w.data.where(w.data['modmax'] >= 64).loc[getConfigColumn(
                w.data).isin([
                    'AllocatedState', 'Freezer', w.best_config, bestScoreConfig
                ])]

            # Scale energies to pJ
            energy_cols = [
                'nvm_leakage (J/Instr)', 'sram_leakage (J/Instr)',
                'sram_read (J/Instr)', 'sram_write(J/Instr)',
                'nvm_read (J/Instr)', 'nvm_write (J/Instr)'
            ]

            df[energy_cols] = df[energy_cols].transform(lambda x: x * 1e12)

            # Print infinities as zero
            df = df.replace(float('inf'), 0)

            # Normalized On+Charge time
            df['total time (s)'] = df['on-time (s)'] + df['charge-time (s)']
            df['normalized total time'] = df['total time (s)'] / df[
                'total time (s)'][0]
            df['normalized on-time'] = df['on-time (s)'] / df[
                'total time (s)'][0]
            df['normalized charge-time'] = df['charge-time (s)'] / df[
                'total time (s)'][0]

            for k in norm_completion_times.keys():
                norm_completion_times[k].append(
                    df.loc[df['dcache config(LW|NL|NS)'] ==
                           k]['normalized total time'].values[0])

            # Replace pipe with dash
            df['dcache config(LW|NL|NS)'].replace(r'\|',
                                                  '-',
                                                  regex=True,
                                                  inplace=True)

            # Emit
            df.to_csv(output_path / f"dcache-epi-{w.name}.csv", index=False)

            # Suspend voltage-drops
            # output one file per workload
            # output the voltage drop, min and max
            df = w.data.loc[w.data['vwarn'] == 2.1, :].loc[getConfigColumn(
                w.data).isin(['AllocatedState', 'Freezer', bestScoreConfig])]

            df['dcache config(LW|NL|NS)'].replace(r'\|',
                                                  '-',
                                                  regex=True,
                                                  inplace=True)

            #df['dcache config(LW|NL|NS)']['2kB (32-2-32)'] = 'MEMIC-MM32'
            df.loc[df['dcache config(LW|NL|NS)'] == '2kB (32-2-32)',
                   'dcache config(LW|NL|NS)'] = 'MEMIC'
            df.loc[df['modmax'] == 32,
                   'dcache config(LW|NL|NS)'] = 'MEMIC-MM32'
            df.loc[df['modmax'] == 16,
                   'dcache config(LW|NL|NS)'] = 'MEMIC-MM16'
            df.loc[df['modmax'] == 8, 'dcache config(LW|NL|NS)'] = 'MEMIC-MM8'

            df['vsuspend drop mean (V)'] = df['vwarn'] - df[
                'v_suspend_mean (V)']
            df['vsuspend drop max diff (V)'] = df['v_suspend_mean (V)'] - df[
                'v_suspend_min (V)']
            df['vsuspend drop min diff (V)'] = df['v_suspend_mean (V)'] - df[
                'v_suspend_max (V)']

            df['esuspend max diff (uJ)'] = df['esuspend_max (uJ)'] - df[
                'esuspend_mean (uJ)']
            df['esuspend min diff (uJ)'] = df['esuspend_min (uJ)'] - df[
                'esuspend_mean (uJ)']

            df.to_csv(output_path / f"vsuspend-{w.name}.csv", index=False)

        # Check the time difference for different modmax
        times = {}
        for w in workloads:
            df = w.data.loc[getConfigColumn(w.data).isin([bestScoreConfig])]
            mm = df['modmax'].values
            times[w.name] = (df['on-time (s)'] + df['charge-time (s)']).values
        score = [0, 0, 0]
        for name, t in times.items():
            score[0] += float(t[0])
            score[1] += float(t[1])
            score[2] += float(t[2])

        print(mm)
        print(times)
        print(score)

    elif 'Instruction' in input_file.name:
        norm_completion_times = {
            'LoadExecute': [],
            'ExecuteInPlace': [],
            bestScoreConfig: []
        }
        for w in workloads:
            df = w.data.loc[getConfigColumn(w.data).isin([
                'ExecuteInPlace', 'LoadExecute', w.best_config, bestScoreConfig
            ])]

            # Scale energies to pJ
            energy_cols = [
                'nvm_leakage (J/Instr)',
                'sram_leakage (J/Instr)',
                'isram_read (J/Instr)',
                'isram_write(J/Instr)',
                'nvm_read (J/Instr)',
            ]

            df[energy_cols] = df[energy_cols].transform(lambda x: x * 1e12)

            # Print infinities as 0
            df = df.replace(float('inf'), 0)

            # Normalized On+Charge time
            df['total time (s)'] = df['on-time (s)'] + df['charge-time (s)']
            df['normalized total time'] = df['total time (s)'] / df[
                'total time (s)'][0]
            df['normalized on-time'] = df['on-time (s)'] / df[
                'total time (s)'][0]
            df['normalized charge-time'] = df['charge-time (s)'] / df[
                'total time (s)'][0]

            for k in norm_completion_times.keys():
                norm_completion_times[k].append(
                    df.loc[df['icache config(LW|NL|NS)'] ==
                           k]['normalized total time'].values[0])

            # Replace pipe with dash
            df['icache config(LW|NL|NS)'].replace(r'\|',
                                                  '-',
                                                  regex=True,
                                                  inplace=True)

            # Emit
            df.to_csv(output_path / f"icache-epi-{w.name}.csv", index=False)
            print(df)
        pass
    else:
        print('Invalid input file')

    print(norm_completion_times)
    for k, v in norm_completion_times.items():
        print(f"{k}, mean={np.mean(v)}, 1-mean={1-np.mean(v)} v={v}")

    exit(0)
