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
LUMINANCE = [600, 800, 1000, 1200, 1400, 1600] # List of luminance values to test at, in Lux
LUMINANCE = [800, 1200, 1600] # List of luminance values to test at, in Lux
INPUT_CAPACITOR = 10.0e-6
OUTPUT_CAPACITOR = 1.0e-6
SIMTIMELIMIT = 90  # seconds
ICACHECFG = (16, 2, 128)
DCACHECFG = (32, 2, 32)
FUSEDBIN = Path("/home/sisl/git/fused/build/fused")
OUTDIR = Path('/tmp/SolarStudy')
WORKLOAD = "nn-gru-logger"
ITERATIONS = 10

def getSensingTime(logfile : str):
    """ Parse the log to find time spent sensing """
    MATCHSTR = r"MemicBoard.accelerometer: @(\d+) ns measurement"
    times = []
    first = True
    valid = True
    with open(logfile, 'r') as infile:
        start = 0
        end = -1

        for line in infile:
            m = re.search(MATCHSTR, line)
            if m:
                if first:
                    start = int(m.group(1))
                    first = False
                else:
                    # End
                    end = int(m.group(1))
                    valid = True # Only valid measurement if more than one line
                continue
            else:
                if valid:
                    times.append(end-start)
                first = True
                valid = False
    return np.array(times)*1e-9

class SolarStudy:
    """
    Fused simulations to get the completion time of workloads while powered by
    the solar cell model at various luminance values.
    """
    def setup(self, workloadPath: str, output_dir: str = OUTDIR):
        self.output_dir = output_dir

        # Create experiments
        self.experiments = []
        for lux in LUMINANCE:
            base = FusedConfig(baseConfig='memic-config.yaml')
            base.set('IoSimulationStopperTarget', ITERATIONS)
            base.set('InputCapacitorValue', INPUT_CAPACITOR)
            base.set('OutputCapacitorValue', OUTPUT_CAPACITOR)
            base.set('LuminanceValue', lux)
            base.set('SimTimeLimit', SIMTIMELIMIT)

            # ------ Baseline config ------
            # I$+AS
            fc = copy.deepcopy(base)
            fc.set('isram.Enable', 'False')
            fc.set('dsram.Enable', 'True')
            fc.set('VoltageWarning', 0.5)
            fc.setDataCacheConfig(0, 0, 0)
            fc.setInstructionCacheConfig(lineWidth=ICACHECFG[0],
                                         nLines=ICACHECFG[1],
                                         nSets=ICACHECFG[2])
            e = Experiment(odir=output_dir,
                           programPath=workloadPath,
                           programName=f'{WORKLOAD}-AS-memic',
                           fusedBinaryPath=FUSEDBIN,
                           fusedConfig=copy.deepcopy(fc))
            self.experiments.append(e)

            # I$ + Freezer
            fc = copy.deepcopy(base)
            fc.set('isram.Enable', 'False')
            fc.set('dsram.Enable', 'True')
            fc.setDataCacheConfig(0, 0, 0)
            fc.set('VoltageWarning', 0.5)
            fc.setInstructionCacheConfig(lineWidth=ICACHECFG[0],
                                         nLines=ICACHECFG[1],
                                         nSets=ICACHECFG[2])
            e = Experiment(odir=output_dir,
                           programPath=workloadPath,
                           programName=f'{WORKLOAD}-FZ-memic',
                           fusedBinaryPath=FUSEDBIN,
                           fusedConfig=copy.deepcopy(fc))
            self.experiments.append(e)

            # MEMIC
            fc = copy.deepcopy(base)
            fc.set('dsram.Enable', 'False')
            fc.set('isram.Enable', 'False')
            fc.set('VoltageWarning', 0.3)
            fc.setInstructionCacheConfig(lineWidth=ICACHECFG[0],
                                         nLines=ICACHECFG[1],
                                         nSets=ICACHECFG[2])
            fc.setDataCacheConfig(lineWidth=DCACHECFG[0],
                                  nLines=DCACHECFG[1],
                                  nSets=DCACHECFG[2])
            e = Experiment(odir=output_dir,
                           programPath=workloadPath,
                           programName=f'{WORKLOAD}-CS-memic',
                           fusedBinaryPath=FUSEDBIN,
                           fusedConfig=copy.deepcopy(fc))
            self.experiments.append(e)

        print(f'SolarStudy generated {len(self.experiments):d}')

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
            ['-'.join(e.name.split('-')[:-3]) for e in self.experiments])

        with open(self.output_dir / 'SolarStudy.csv', 'w+') as of:
            for app in apps:
                header = f'{app},,,,,,,,,,,,,,,,\n'\
                    f'icache config(LW|NL|NS), luminance (lux), on-time (s), charge-time (s), sense-time (s), '\
                    f'mean completion-time (s), stdev completion time (s), '\
                    f'code size (kB), data size (kB), capacitance (F), '\
                    f'vwarn, v_suspend_mean (V), v_suspend_min (V), '\
                    f'v_suspend_max (V), '\
                    f'tsuspend_mean (us), tsuspend_min (us), tsuspend_max (us), tsuspend_tot (s), ' \
                    f'testdir, hash\n'
                of.write(header)

                for e in filter(lambda x: app in x.name, self.experiments):
                    icfg = e.instructionCacheConfig
                    iCacheSize = np.prod(icfg)

                    # Instruction config
                    if (iCacheSize == 0):  # SRAM or XiP
                        configStr = 'LoadExecute' if 'sram' in e.name else 'ExecuteInPlace'
                    else:  # Cache
                        configStr = f'{iCacheSize//1024:d}kB ({icfg[0]:d}|{icfg[1]:d}|{icfg[2]:d})'

                    # Data config
                    if '-AS-' in e.name:
                        configStr += '-AllocatedState'
                    elif '-FZ-' in e.name:
                        configStr += '-Freezer'
                    else:  # Cache
                        dcfg = e.dataCacheConfig
                        dCacheSize = np.prod(dcfg)
                        configStr += f'-{dCacheSize//1024:d}kB ({dcfg[0]:d}|{dcfg[1]:d}|{dcfg[2]:d})'

                    if e.failed:
                        of.write(configStr + ',' + ','.join(
                            ['inf'
                             for _ in header.split('\n')[1].split(',')][:-3]
                        ) + f", {e.fusedConfig.get('InputCapacitorValue')}, {e.hash:08x}\n"
                                 )
                        continue

                    e.loadSummary()

                    mean_completion = np.mean(e.completion_times)
                    std_completion = np.std(e.completion_times)
                    vsuspend = e.suspendVoltages
                    #esuspend = e.suspendEnergies * 1e9
                    vwarn = e.fusedConfig.get('VoltageWarning')
                    tsuspend = e.suspendTimes * 1e6

                    chargetime = e.runtime - e.ontime
                    ontime = e.ontime
                    if math.isnan(chargetime) or math.isinf(chargetime):
                        # inf-inf from above results in nan
                        chargetime = float('inf')
                        ontime = float('inf')

                    luminance = e.fusedConfig.get('LuminanceValue')

                    sensetime = getSensingTime(e.testdir / 'stdout.log')

                    of.write(
                        f'{configStr}, {luminance}, {ontime}, {chargetime}, {sensetime.sum()}, '
                        f'{mean_completion}, {std_completion}, '
                        f'{e.codesize / 1024:.2f}, {e.datasize / 1024 :.2f}, '
                        f'{e.fusedConfig.get("InputCapacitorValue")}, '
                        f'{vwarn}, {vsuspend["v_in"].mean()}, {vsuspend["v_in"].min()}, '
                        f'{vsuspend["v_in"].max()}, '
                        f'{tsuspend.mean()}, {tsuspend.min()}, {tsuspend.max()}, {tsuspend.sum()*1e-6}, '
                        f'{e.testdir.name}, {e.hash:08x}\n')

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
            print(f"Starting {e.name}-{e.hash:08x}")
            e.run(force=False)
            retval = '{:s}, cap=i{}/o{} F, l={} lux, vwarn={:.2f}, i{}/d{} (hash={:08x})... '\
                    .format(e.name,
                            e.fusedConfig.get('InputCapacitorValue'),
                            e.fusedConfig.get('OutputCapacitorValue'),
                            e.fusedConfig.get('LuminanceValue'),
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
        ./SolarStudy.py <workloadBinPath>
        """)
        exit(1)

    ics = SolarStudy()
    ics.setup(sys.argv[1])
    ics.run()
    ics.emit()
    exit(0)
