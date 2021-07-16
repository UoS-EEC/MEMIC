#!/usr/bin/env python3

import code
import copy
import datetime
import json
import math
import time
import os
import re
import shutil
import subprocess as sp
import sys
from pathlib import Path

sys.path.append('lib')

import Experiment
import Factorize
import numpy as np
import pandas as pd
from Experiment import Experiment
from FusedConfig import FusedConfig

# Parameters
TARGET_ONTIME_PER_WORKLOAD = 5.0  # seconds
OUTDIR = Path('/tmp/DataCacheStudy')
FUSEDBIN = Path("~/git/fused/build/fused")


class DataCacheStudy:
    """
    Fused simulations to explore the use of data caches

    Runs with default instruction cache configuration, and sweeps data cache.

    The point of this study is to find out whether it's best to load data from
    NVM to a separate SRAM, or to use a data cache on a system that runs
    intermittently on a very constrained power source.
    """
    def setup(self, workloadPath: str, output_dir: str = OUTDIR):
        self.output_dir = output_dir

        # Find workloads
        workloads = set([
            '-'.join(f.stem.split('-')[:-2])
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

        #configs = configs[::20]

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
            if app in COMPLETION_TIMES.keys():
                iterations = math.ceil(
                    TARGET_ONTIME_PER_WORKLOAD / COMPLETION_TIMES[app]) + 1
                print(f"app {app} iterations: {iterations}")
            else:
                iterations = 1 + 1
                print(f"app {app} iterations: {iterations} (default)")

            # ------ Baseline configs ------

            # AllocatedState (keep data in SRAM)
            fc = FusedConfig(baseConfig='memic-config.yaml')
            fc.set('IoSimulationStopperTarget', iterations)
            fc.set('isram.Enable', 'False')
            fc.set('dsram.Enable', 'True')
            fc.setDataCacheConfig(0, 0, 0)
            e = Experiment(odir=output_dir,
                           programPath=workloadPath,
                           programName=f'{app}-AS-memic',
                           fusedBinaryPath=FUSEDBIN,
                           fusedConfig=copy.deepcopy(fc))
            self.experiments.append(e)

            # Freezer
            fc = FusedConfig(baseConfig='memic-config.yaml')
            fc.set('IoSimulationStopperTarget', iterations)
            fc.set('isram.Enable', 'False')
            fc.set('dsram.Enable', 'True')
            fc.setDataCacheConfig(0, 0, 0)
            e = Experiment(odir=output_dir,
                           programPath=workloadPath,
                           programName=f'{app}-FZ-memic',
                           fusedBinaryPath=FUSEDBIN,
                           fusedConfig=copy.deepcopy(fc))
            self.experiments.append(e)

            # ------ Cache exploration ------
            for cfg in configs:
                fc = FusedConfig(baseConfig='memic-config.yaml')
                fc.set('isram.Enable', 'False')
                fc.set('dsram.Enable', 'False')
                fc.set('IoSimulationStopperTarget', iterations)
                fc.setDataCacheConfig(lineWidth=cfg[0],
                                      nLines=cfg[1],
                                      nSets=cfg[2])
                e = Experiment(odir=output_dir,
                               programPath=workloadPath,
                               programName=f'{app}-CS-memic',
                               fusedBinaryPath=FUSEDBIN,
                               fusedConfig=copy.deepcopy(fc))
                self.experiments.append(e)

            # ------ Low vwarn, limited modified state ------
            for mm in [32, 16]:
                cfg = (32, 2, 32)
                fc = FusedConfig(baseConfig='memic-config.yaml')
                fc.set('isram.Enable', 'False')
                fc.set('dsram.Enable', 'False')
                fc.set('IoSimulationStopperTarget', iterations)
                fc.set('dcache_ctrl.MaxModified', mm)
                fc.setDataCacheConfig(lineWidth=cfg[0],
                                      nLines=cfg[1],
                                      nSets=cfg[2])
                e = Experiment(odir=output_dir,
                               programPath=workloadPath,
                               programName=f'{app}-CS-memic',
                               fusedBinaryPath=FUSEDBIN,
                               fusedConfig=copy.deepcopy(fc))
                self.experiments.append(e)

        print(f'DataCacheStudy generated {len(self.experiments):d} '
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
        p1 = Pool(7)  # 7 Workers for running simulations
        #p2 = Pool(1)  # 1 worker for parsing results

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
            ['-'.join(e.name.split('-')[:-2]) for e in self.experiments])

        with open(self.output_dir + '/DataCacheStudy.csv', 'w+') as of:
            for app in apps:
                header = f"{app},,,,,,,,,,,,,,,,,,,,,,,\n"\
                         'dcache config(LW|NL|NS), nvm_leakage (J/Instr), '\
                         'sram_leakage (J/Instr), sram_read (J/Instr), '\
                         'sram_write(J/Instr),  nvm_read (J/Instr), '\
                         'nvm_write (J/Instr), on-time (s), charge-time (s), '\
                         'mean completion-time (s), stdev completion time (s), '\
                         'vwarn, v_suspend_mean (V), v_suspend_min (V), '\
                         'v_suspend_max (V), '\
                         'esuspend_mean (uJ), esuspend_min (uJ), esuspend_max (uJ), ' \
                         'tsuspend_mean (us), tsuspend_min (us), tsuspend_max (us), ' \
                         'modmax, power cycles, testdir, hash\n'
                of.write(header)
                for e in filter(lambda x: app in x.name, self.experiments):

                    if '-AS-' in e.name:
                        configStr = 'AllocatedState'
                    elif '-FZ-' in e.name:
                        configStr = 'Freezer'
                    else:  # Cache
                        dcfg = e.dataCacheConfig
                        dCacheSize = np.prod(dcfg)
                        configStr = f'{dCacheSize//1024:d}kB ({dcfg[0]:d}|{dcfg[1]:d}|{dcfg[2]:d})'

                    if e.failed:
                        of.write(configStr + ',' + ','.join(
                            ['inf'
                             for _ in header.split('\n')[1].split(',')][:-3]) +
                                 f",{e.testdir}, {e.hash}\n")
                        continue

                    e.loadSummary()

                    #if (dCacheSize == 0):  # AllocatedState / Freezer
                    if '-AS-' in e.name or '-FZ-' in e.name:
                        sram_leakage = e.get_static_group_epi('dsram')
                        dsram_read = e.get_group_epi('dsram',
                                                     'read',
                                                     exclude='ctrl')
                        dsram_write = e.get_group_epi('dsram',
                                                      'write',
                                                      exclude='ctrl')
                    else:  # Cache
                        sram_leakage = e.get_static_epi('dcache')
                        dsram_read = e.get_epi(
                            'dcache bytes read') + e.get_epi(
                                'dcache tag read bits')
                        dsram_write = e.get_epi(
                            'dcache bytes written') + e.get_epi(
                                'dcache tag write bits')

                    nvm_read = e.get_group_epi('dnvm', 'read', exclude='ctrl')
                    nvm_write = e.get_group_epi('dnvm',
                                                'write',
                                                exclude='ctrl')

                    nvm_leakage = e.get_static_group_epi('dnvm')
                    vsuspend = e.suspendVoltages
                    esuspend = e.suspendEnergies * 1e6
                    tsuspend = e.suspendTimes * 1e6
                    vwarn = e.fusedConfig.get('VoltageWarning')
                    modmax = e.fusedConfig.get('dcache_ctrl.MaxModified')

                    if esuspend.mean() > 2:
                        print(
                            f"{e.testdir} weird suspend energy: pwrcycles={e.nPowerCycles}, esuspend={esuspend}, tsuspend={tsuspend}"
                        )

                    chargetime = e.runtime - e.ontime
                    ontime = e.ontime
                    mean_completion = np.mean(e.completion_times)
                    std_completion = np.std(e.completion_times)
                    if math.isnan(chargetime) or math.isinf(chargetime):
                        # inf-inf from above results in nan
                        chargetime = float('inf')
                        ontime = float('inf')

                    of.write(
                        f'{configStr}, '
                        f'{nvm_leakage}, '
                        f'{sram_leakage}, '
                        f'{dsram_read}, {dsram_write}, {nvm_read}, '
                        f'{nvm_write}, {ontime}, {chargetime},'
                        f'{mean_completion}, {std_completion}, '
                        f'{vwarn}, {vsuspend.mean()}, {vsuspend.min()}, '
                        f'{vsuspend.max()}, '
                        f'{esuspend.mean()}, {esuspend.min()}, {esuspend.max()}, '
                        f'{tsuspend.mean()}, {tsuspend.min()}, {tsuspend.max()}, '
                        f'{modmax}, {e.nPowerCycles}, {e.testdir.name}, {e.hash:08x}\n'
                    )

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
        return (linewidth >= 8) and (linewidth <= 64) and (nlines <= 16)

    def runExperiment(self, e: Experiment):
        """
        Run a single experiment and  report a result summary, then compress
        traces.
        """
        print('{} started'.format(e.name))
        e.run(force=False)
        #print('{} finished'.format(e.name))
        retval = '{:s}, cap={} F, p={} W, vwarn={:.2f}, i{}/d{} (hash={:08x})... '\
                .format(e.name,
                        e.fusedConfig.get('CapacitorValue'),
                        e.fusedConfig.get('PowerSupplyPower'),
                        e.fusedConfig.get('VoltageWarning'),
                        e.instructionCacheConfig,
                        e.dataCacheConfig,
                        e.hash)

        try:
            e.parseResults()
        except Exception as exc:
            print(exc)
            (e.testdir / Path('summary.txt')).touch()

        e.compressTraces()
        retval += f't={e.runtime:.6f} s'
        return retval

    def parseExperiments(self, experiments):
        print("parseExperiments")
        while np.any(not e.parsed and not e.failed for e in experiments):
            for e in filter(lambda x: not x.parsed and x.completed,
                            experiments):
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
        ./DataCacheStudy.py <workloadBinPath>
        """)
        exit(1)
    ics = DataCacheStudy()
    ics.setup(sys.argv[1])
    ics.run()
    ics.emit()
    exit(0)
