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
import logging

sys.path.append('lib')

import Experiment
import Factorize
import numpy as np
import pandas as pd
from Experiment import Experiment
from FusedConfig import FusedConfig

# Parameters
TARGET_ONTIME_PER_WORKLOAD = 5.0  # seconds
ICACHECFG = (16, 2, 128)
DCACHECFG = (32, 2, 32)
CAPACITOR = 10.0e-6  # Farad
FUSEDBIN = Path("~/git/fused/build/fused")
OUTDIR = Path('/tmp/CompletionTimes')


class FaseStudy:
    """
    Fused simulations to run the logging sensor node application with
    Failure-Atomic Sections (FASE) on different configurations.

    The main point of this study is to compare FASE support between
    AllocatedState, Freezer and MEMIC.
    """
    def setup(self, workloadPath: str, output_dir: str = OUTDIR):
        self.output_dir = output_dir

        # Find workloads
        workloads = set([
            '-'.join(f.stem.split('-')[:-2])
            for f in Path(workloadPath).glob('*.elf')
        ])
        print(workloads)

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
            # ICache+AS
            fc = FusedConfig(baseConfig='memic-config.yaml')
            fc.set('isram.Enable', 'False')
            fc.set('dsram.Enable', 'True')
            fc.set('IoSimulationStopperTarget', iterations)
            fc.set('CapacitorValue', CAPACITOR)
            fc.setDataCacheConfig(0, 0, 0)
            fc.setInstructionCacheConfig(lineWidth=ICACHECFG[0],
                                         nLines=ICACHECFG[1],
                                         nSets=ICACHECFG[2])
            e = Experiment(odir=output_dir,
                           programPath=workloadPath,
                           programName=f'{app}-AS-memic',
                           fusedBinaryPath=FUSEDBIN,
                           fusedConfig=copy.deepcopy(fc))
            self.experiments.append(e)

            # Cache + Freezer
            fc = FusedConfig(baseConfig='memic-config.yaml')
            fc.set('IoSimulationStopperTarget', iterations)
            fc.set('isram.Enable', 'False')
            fc.set('dsram.Enable', 'True')
            fc.set('CapacitorValue', CAPACITOR)
            fc.setDataCacheConfig(0, 0, 0)
            fc.setInstructionCacheConfig(lineWidth=ICACHECFG[0],
                                         nLines=ICACHECFG[1],
                                         nSets=ICACHECFG[2])
            e = Experiment(odir=output_dir,
                           programPath=workloadPath,
                           programName=f'{app}-FZ-memic',
                           fusedBinaryPath=FUSEDBIN,
                           fusedConfig=copy.deepcopy(fc))
            self.experiments.append(e)

            # MEMIC, three different UndoLogger sizes
            for ulSize in [2, 4, 8]:
                fc = FusedConfig(baseConfig='memic-config.yaml')
                fc.set('IoSimulationStopperTarget', iterations)
                fc.set('dsram.Enable', 'False')
                fc.set('isram.Enable', 'False')
                fc.set('CapacitorValue', CAPACITOR)
                fc.set('MemicBoard.mcu.undoLogger.capacity', ulSize)

                fc.setInstructionCacheConfig(lineWidth=ICACHECFG[0],
                                             nLines=ICACHECFG[1],
                                             nSets=ICACHECFG[2])
                fc.setDataCacheConfig(lineWidth=DCACHECFG[0],
                                      nLines=DCACHECFG[1],
                                      nSets=DCACHECFG[2])
                e = Experiment(odir=output_dir,
                               programPath=workloadPath,
                               programName=f'{app}-CS-memic',
                               fusedBinaryPath=FUSEDBIN,
                               fusedConfig=copy.deepcopy(fc))
                self.experiments.append(e)

                # Binary that does not use unsafe zone
                fc = FusedConfig(baseConfig='memic-config.yaml')
                fc.set('IoSimulationStopperTarget', iterations)
                fc.set('dsram.Enable', 'False')
                fc.set('isram.Enable', 'False')
                fc.set('CapacitorValue', CAPACITOR)
                fc.set('MemicBoard.mcu.undoLogger.capacity', ulSize)

                fc.setInstructionCacheConfig(lineWidth=ICACHECFG[0],
                                             nLines=ICACHECFG[1],
                                             nSets=ICACHECFG[2])
                fc.setDataCacheConfig(lineWidth=DCACHECFG[0],
                                      nLines=DCACHECFG[1],
                                      nSets=DCACHECFG[2])
                e = Experiment(odir=output_dir,
                               programPath=workloadPath,
                               programName=f'{app}-CSnounsafe-memic',
                               fusedBinaryPath=FUSEDBIN,
                               fusedConfig=copy.deepcopy(fc))
                self.experiments.append(e)

        print(f'FaseStudy generated {len(self.experiments):d} '
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

        pool = Pool()  # Creates a pool of n_cpus workers
        for n, res in enumerate(
                pool.imap_unordered(self.runExperiment, self.experiments)):
            print(f'{n+1:d}/{len(self.experiments):d} {res}')

    def emit(self):
        if self.success is None:
            raise Exception('Not run yet, run first')
        if self.success == False:
            raise Exception("Failed run, won't emit")

        apps = set(
            ['-'.join(e.name.split('-')[:-2]) for e in self.experiments])

        with open(self.output_dir + '/FaseStudy.csv', 'w+') as of:
            for app in apps:
                header = f'{app},,,,,\n'\
                    f'icache config(LW|NL|NS), on-time (s), charge-time (s), '\
                    f'mean completion-time (s), stdev completion time (s), '\
                    f'code size (kB), data size (kB), UndoLogger.capacity, hash\n'
                of.write(header)

                for e in filter(lambda x: app in x.name, self.experiments):
                    icfg = e.instructionCacheConfig
                    iCacheSize = np.prod(icfg)

                    # Data config
                    if '-AS-' in e.name:
                        configStr = '-AllocatedState'
                    elif '-FZ-' in e.name:
                        configStr = '-Freezer'
                    else:  # Cache
                        dcfg = e.dataCacheConfig
                        dCacheSize = np.prod(dcfg)
                        configStr = f'-{dCacheSize//1024:d}kB ({dcfg[0]:d}|{dcfg[1]:d}|{dcfg[2]:d})'

                    if 'nounsafe' in e.name:
                        configStr += '-nounsafe'

                    if e.failed:
                        of.write(configStr + ',' + ','.join(
                            ['inf'
                             for _ in header.split('\n')[1].split(',')][:-2]) +
                                 f",{e.hash}\n")
                        continue

                    e.loadSummary()

                    mean_completion = np.mean(e.completion_times)
                    std_completion = np.std(e.completion_times)

                    chargetime = e.runtime - e.ontime
                    ontime = e.ontime
                    if math.isnan(chargetime) or math.isinf(chargetime):
                        # inf-inf from above results in nan
                        chargetime = float('inf')
                        ontime = float('inf')

                    of.write(
                        f'{configStr}, {ontime}, {chargetime}, '
                        f'{mean_completion}, {std_completion}, '
                        f'{e.codesize / 1024:.2f}, {e.datasize / 1024 :.2f}, '
                        f"{e.fusedConfig.get('MemicBoard.mcu.undoLogger.capacity')}, "
                        f'{e.hash:08x}\n')

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
            print(f"Staring {e.name}-{e.hash:08x}")
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
            retval = f"{e.name}-{e.hash:08x} already completed"

        try:
            e.parseResults()
        except Exception as exc:
            print(exc)

        e.deleteFile('cpu.vcd')
        e.compressTraces()
        return retval


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("""
        Usage:
        ./FaseStudy.py <workloadBinPath>
        """)
        exit(1)

    ics = FaseStudy()
    ics.setup(sys.argv[1])
    ics.run()
    ics.emit()
    exit(0)
