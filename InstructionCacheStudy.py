#!/usr/bin/env python3

import code
import copy
import datetime
import json
import math
import os
import re
import shutil
import subprocess as sp
import sys
from pathlib import Path
import time

sys.path.append('lib')

import Experiment
import Factorize
import numpy as np
import pandas as pd
from Experiment import Experiment
from FusedConfig import FusedConfig

# Parameters
TARGET_ONTIME_PER_WORKLOAD = 5.0  # seconds
FUSEDBIN = Path("~/git/fused/build/fused")
OUTDIR = Path('/tmp/CompletionTimes')


class InstructionCacheStudy:
    """
    Fused simulations to explore the use of instruction caches

    Runs with data cache disabled, and sweeps the instruction cache
    organization/size.

    The main point of this study is to find out what's most energy-efficient
    between:
        - XiP: Running instructions directly from NVM
        - LoadExecute: Load instructions into SRAM then execute
        - Instruction cache

    """
    def setup(self, workloadPath: str, output_dir: str = OUTDIR):
        self.output_dir = output_dir

        # Find workloads
        workloads = set([
            '-'.join(f.stem.split('-')[:-3])
            for f in Path(workloadPath).glob('*.elf')
        ])
        print(workloads)

        # Cache configurations to try
        # Generate all valid & reasonable 1kB, 2kB, 4kB and 8kB cache configs
        configs = [
            v for sizekb in [1, 2, 4, 8]
            for v in Factorize.factor3(sizekb * 1024)
            if self.isReasonableConfig(v)
        ]

        #configs = configs[::4]

        print(f'Generated {len(configs):d} reasonable cache configurations')

        # Load workload nominal completion times
        with open('appRunTimes.csv', 'r') as infile:
            COMPLETION_TIMES = {
                line.split(', ')[0]: float(line.split(', ')[1])
                for line in infile.readlines()[1:]
            }

        # Create experiments
        self.experiments = []
        for app in workloads:
            # Target iterations
            iterations = math.ceil(
                TARGET_ONTIME_PER_WORKLOAD / COMPLETION_TIMES[app]) + 1
            print(f"app {app} iterations: {iterations}")

            # ------ Baseline config ------
            # Execute in Place (XiP) (no instruction cache)
            fc = FusedConfig(baseConfig='memic-config.yaml')
            fc.set('IoSimulationStopperTarget', iterations)
            fc.setDataCacheConfig(0, 0, 0)
            fc.setInstructionCacheConfig(0, 0, 0)
            fc.set('isram.Enable', 'False')
            fc.set('dsram.Enable', 'True')
            e = Experiment(odir=output_dir,
                           programPath=workloadPath,
                           programName=f'{app}-AS-memic-cache',
                           fusedBinaryPath=FUSEDBIN,
                           fusedConfig=copy.deepcopy(fc))
            self.experiments.append(e)

            # LoadExecute (no instruction cache)
            fc = FusedConfig(baseConfig='memic-config.yaml')
            fc.set('IoSimulationStopperTarget', iterations)
            fc.setDataCacheConfig(0, 0, 0)
            fc.setInstructionCacheConfig(0, 0, 0)
            fc.set('isram.Enable', 'True')
            fc.set('dsram.Enable', 'True')
            e = Experiment(odir=output_dir,
                           programPath=workloadPath,
                           programName=f'{app}-AS-memic-sram',
                           fusedBinaryPath=FUSEDBIN,
                           fusedConfig=copy.deepcopy(fc))
            self.experiments.append(e)

            for cfg in configs:
                # Load default config
                fc = FusedConfig(baseConfig='memic-config.yaml')
                fc.set('IoSimulationStopperTarget', iterations)
                fc.set('dsram.Enable', 'True')
                fc.setDataCacheConfig(0, 0, 0)
                fc.setInstructionCacheConfig(lineWidth=cfg[0],
                                             nLines=cfg[1],
                                             nSets=cfg[2])
                e = Experiment(odir=output_dir,
                               programPath=workloadPath,
                               programName=f'{app}-AS-memic-cache',
                               fusedBinaryPath=FUSEDBIN,
                               fusedConfig=copy.deepcopy(fc))
                self.experiments.append(e)

        print(f'InstructionCacheStudy generated {len(self.experiments):d} '
              f'experiments from {len(workloads):d} workloads')

        # Create output directories
        for e in self.experiments:
            e.createTestDir()

        # Export this script to output dir (for reproducibility)
        shutil.copy(Path(__file__), Path(output_dir))

        self.setupComplete = True

    def run(self):
        """
        Run all experiments
        """
        if not self.setupComplete:
            raise Exception('Not set up yet, run setup first')
        self.success = True

        print(f'------ Running {len(self.experiments):d} experiments ------')
        # Run Experiments
        from multiprocessing import Pool, TimeoutError

        # Run experiments
        p1 = Pool(7)  # Creates a pool of n_cpus workers
        #p2 = Pool(1)

        r1 = p1.map_async(self.runExperiment, self.experiments)
        #r2 = p2.map_async(self.parseExperiments, [self.experiments])

        r1.get()
        #r2.get()

    def emit(self):
        if self.success is None:
            raise Exception('Not run yet, run first')
        if self.success == False:
            raise Exception("Failed run, won't emit")

        apps = set(
            ['-'.join(e.name.split('-')[:-1]) for e in self.experiments])

        with open(self.output_dir + '/InstructionCacheStudy.csv', 'w+') as of:
            for app in apps:
                header = f'{app},,,,,,,,,,,\n'\
                    f'icache config(LW|NL|NS), nvm_leakage (J/Instr), '\
                    f'sram_leakage (J/Instr), isram_read (J/Instr), '\
                    f'isram_write(J/Instr),  nvm_read (J/Instr), '\
                    f'on-time (s), charge-time (s), '\
                    f'mean completion-time (s), stdev completion time (s), '\
                    f'power cycles,'\
                    f'hash\n'
                of.write(header)

                for e in filter(lambda x: app in x.name, self.experiments):
                    icfg = e.instructionCacheConfig
                    iCacheSize = np.prod(icfg)

                    if (iCacheSize == 0):  # SRAM or XiP
                        configStr = 'LoadExecute' if 'sram' in e.name else 'ExecuteInPlace'
                    else:  # Cache
                        configStr = f'{iCacheSize//1024:d}kB ({icfg[0]:d}|{icfg[1]:d}|{icfg[2]:d})'

                    if e.failed:
                        of.write(configStr + ',' + ','.join(
                            ['inf'
                             for _ in header.split('\n')[1].split(',')][:-2]) +
                                 f",{e.hash}\n")
                        continue

                    e.loadSummary()

                    if (iCacheSize == 0):  # SRAM or XiP
                        configStr = 'LoadExecute' if 'sram' in e.name else 'ExecuteInPlace'
                        sram_leakage = e.get_static_group_epi('isram')
                        isram_read = e.get_group_epi('isram',
                                                     'read',
                                                     exclude='ctrl')
                        isram_write = e.get_group_epi('isram',
                                                      'write',
                                                      exclude='ctrl')
                    else:  # Cache
                        configStr = f'{iCacheSize//1024:d}kB ({icfg[0]:d}|{icfg[1]:d}|{icfg[2]:d})'
                        sram_leakage = e.get_static_group_epi('icache')
                        isram_read = e.get_epi(
                            'icache bytes read') + e.get_epi(
                                'icache tag read bits')
                        isram_write = e.get_epi(
                            'icache bytes written') + e.get_epi(
                                'icache tag write bits')

                    nvm_read = e.get_group_epi('invm', 'read', exclude='ctrl')
                    nvm_leakage = e.get_static_group_epi('invm')

                    mean_completion = np.mean(e.completion_times)
                    std_completion = np.std(e.completion_times)

                    chargetime = e.runtime - e.ontime
                    ontime = e.ontime
                    if math.isnan(chargetime) or math.isinf(chargetime):
                        # inf-inf from above results in nan
                        chargetime = float('inf')
                        ontime = float('inf')

                    of.write(
                        f'{configStr}, {nvm_leakage}, {sram_leakage}, '
                        f'{isram_read}, {isram_write}, {nvm_read}, {ontime}, '
                        f'{chargetime}, {mean_completion}, {std_completion}, '
                        f'{e.nPowerCycles}, {e.hash:08x}\n')

    def plot(self):
        if self.success is None:
            raise Exception('Not run yet, run first')
        if self.success == False:
            raise Exception("Failed run, won't emit")

    def isReasonableConfig(self, cfg: tuple):
        """
        Check if the configuration of (linewidth, nlines, nsets) is reasonable
        """
        linewidth, nlines, _ = cfg
        return (linewidth >= 8) and (linewidth <= 64) and (nlines <= 4)

    def runExperiment(self, e: Experiment):
        """
        Run a single experiment and  report a result summary, then compress
        traces.
        """
        if not e.completed:
            print(f"running {e.name}, testdir={e.testdir}")
            e.run(force=False)
            retval = '{:s}, cap={} F, p={} W, vwarn={:.2f}, i{}/d{} (hash={:08x})... '\
                    .format(e.name,
                            e.fusedConfig.get('CapacitorValue'),
                            e.fusedConfig.get('PowerSupplyPower'),
                            e.fusedConfig.get('VoltageWarning'),
                            e.instructionCacheConfig,
                            e.dataCacheConfig,
                            e.fusedConfig.hash)

            retval += f't={e.runtime:.6f} s'
        else:
            retval = f"{e.name} already completed"

        try:
            e.parseResults()
        except Exception as exc:
            print(exc)
            (e.testdir / Path('summary.txt')).touch()

        e.compressTraces()
        return retval

    def parseExperiments(self, experiments):
        while np.any(not e.parsed and not e.failed for e in experiments):
            for e in filter(lambda x: not x.parsed and x.completed,
                            experiments):
                print(f"parsing {e.name}, testdir={e.testdir}")
                if not e.completed:
                    raise Exception(
                        f"{e.name} has not completed, testdir= {e.testdir}")
                try:
                    e.parseResults()
                except Exception as exc:
                    print(exc)
                    (e.testdir / Path('summary.txt')).touch()

                e.compressTraces()
            else:
                if np.all([e.completed or e.failed for e in experiments]):
                    print("Finished parsing (for loop)")
                    return True
            time.sleep(0.5)
        print("Finished parsing (while loop)")
        return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("""
        Usage:
        ./InstructionCacheStudy.py <workloadBinPath>
        """)
        exit(1)
    ics = InstructionCacheStudy()
    ics.setup(sys.argv[1])
    ics.run()
    ics.emit()
    exit(0)
