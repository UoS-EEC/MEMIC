#!/usr/bin/env python3
import code
import math
import os
import re
import shutil
import subprocess as sp
import sys
import zlib
from pathlib import Path
from typing import Sequence
import yaml

import numpy as np
import pandas as pd
from FusedConfig import FusedConfig

if not sys.version_info.major >= 3:
    print("Python 3 is required, but you are using Python {}.{}".format(
        sys.version_info.major, sys.version_info.minor))
    sys.exit(1)

NVM_SIZE = 32 * 1024
ADDRESS_BITS = int(math.log2(NVM_SIZE))


class Experiment:
    """ Utility class to set up, run, and analyse experiments """
    def __init__(self, odir: str, programPath: str, programName: str,
                 fusedBinaryPath: str, fusedConfig: FusedConfig):
        self.programHexPath = Path(programPath) / Path(programName + '.ihex')
        self.programElfPath = Path(programPath) / Path(programName + '.elf')
        self.fusedConfig = fusedConfig
        self.hash = self.fusedConfig.hash
        self.name = programName
        self.testdir = Path(odir) / '{}-{:08x}'.format(programName, self.hash)
        self.fusedBinary = Path(fusedBinaryPath)

        self.results = {}  # Dictionary to store results after parsing

        if not self.programHexPath.exists():
            raise ValueError('Hex file {} missing'.format(self.programHexPath))
        if not self.programElfPath.exists():
            raise ValueError('ELF file {} missing'.format(self.programHexPath))
        if not self.fusedBinary.exists():
            raise ValueError('Fused binary {} missing'.format(
                self.fusedBinary))

    def createTestDir(self):
        try:
            self.testdir.mkdir(parents=True)
        except FileExistsError:
            pass

        shutil.copy(self.programHexPath, self.testdir)
        shutil.copy(self.programElfPath, self.testdir)

        if self.fusedConfig.get('BootTracePath') != 'none':
            shutil.copy(Path(self.fusedConfig.get('BootTracePath')),
                        self.testdir)
            self.fusedConfig.set(
                'BootTracePath', self.testdir /
                Path(self.fusedConfig.get('BootTracePath')).name)

        if self.fusedConfig.get('VccMultiplierPath') != 'none':
            shutil.copy(Path(self.fusedConfig.get('VccMultiplierPath')),
                        self.testdir)
            self.fusedConfig.set(
                'VccMultiplierPath', self.testdir /
                Path(self.fusedConfig.get('VccMultiplierPath')).name)

        self.fusedConfig.emit(self.testdir)

    def run(self, force=False):
        """ Create simulation directory and run simulation """

        # Check if test already done
        if (self.completed or self.failed) and (not force):
            self.checkLogsForFails()
            return

        if Path.exists(self.testdir):
            # Delete old traces if they exist
            for p in self.testdir.glob('*.csv'):
                print(f"Removing old trace file {p}")
                p.unlink()
        else:
            # Create test dir
            self.createTestDir()

        cmd = [
            '{}'.format(self.fusedBinary), '--program',
            str(self.testdir / Path(self.name + '.ihex')), '--odir',
            str(self.testdir), '--config',
            str(self.testdir / Path('config.yaml'))
        ]

        with open(self.testdir / Path('cmd.txt'), 'w+') as of:
            of.write(' '.join(cmd))

        fused_stderr = open(self.testdir / Path('stderr.log'), 'w+')
        fused_stdout = open(self.testdir / Path('stdout.log'), 'w+')
        fusedRun = sp.run(cmd, stdout=fused_stdout, stderr=fused_stderr)
        fused_stderr.close()
        fused_stdout.close()

        if fusedRun.returncode != 0:
            with open(self.testdir / Path('stdout.log'), 'r') as infile:
                (self.testdir / Path('failed')).touch()
                content = infile.read()
                if 'SW_TEST_FAIL' in content:
                    print("SW_TEST_FAIL at {}".format(self.testdir.name))
                else:
                    print(f"Fused exited with an error.\n\t See  "
                          f"{(self.testdir / Path('stdout.log'))}")

        (self.testdir / Path('done')).touch()  # Mark folder as done
        #print("Experiment::run {} run done\n".format(self.name))
        return

    def checkLogsForFails(self) -> bool:
        with open(self.testdir / Path('stdout.log'), 'r') as infile:
            content = infile.read()

        vsuspendMin = np.min(self.suspendVoltages)
        if vsuspendMin < self.fusedConfig.get("CpuCoreVoltage") + 0.02:
            print(f"{self.name} failed due to a "
                  f"failed checkpoint with "
                  f"vcc={vsuspendMin:.2f} V (testdir={self.testdir})")
            #(self.testdir / Path('failed')).touch()
            return True
        else:
            return False

    def parseResults(self):
        print(f'Parsing results from {self.name}-{self.hash:08x}')

        if not self.completed:  # or self.failed:
            raise Exception(f'{self.name} has not completed.')
        elif self.failed:
            raise Exception(f'{self.name}-{self.hash:08x} has failed.')
        elif Path.exists(self.testdir / 'summary.txt'):
            self.loadSummary()  # Load summary from file
        else:
            include_features = [
                'time(us)', 'invm bytes read', 'invm bytes written',
                'dnvm bytes read', 'dnvm bytes written', 'isram bytes read',
                'isram bytes written', 'dsram bytes read',
                'dsram bytes written', 'CPU n instructions'
            ]
            # Parse eventlog
            if self.dataCacheEnabled:
                include_features += [
                    'dcache bytes read', 'dcache bytes written',
                    'dcache tag access bits'
                ]
            if self.instructionCacheEnabled:
                include_features += [
                    'icache bytes read', 'icache bytes written',
                    'icache tag access bits'
                ]

            paths = [
                self.testdir / 'powerModelChannel_eventlog.csv',
                self.testdir / 'powerModelChannel_eventlog.csv.gz'
            ]

            if not np.any([p.exists() for p in paths]):
                raise Exception(f"Event log missing from {paths}")

            for path in paths:
                try:
                    if path.exists():
                        elog = pd.read_csv(path)
                        break
                except Exception as e:
                    raise Exception(
                        f"{e}\n ...Encountered while reading file {path}")

            # Filter out unnecessary features
            """
            include_features = [
                self.expandHierarchicalName(list(elog.columns), v)
                for v in include_features
            ]
            elog = elog[include_features]
            """

            # Adjust time
            elog['time(us)'] = elog['time(us)'] * 1.0e6

            get_col = lambda name: \
                    elog[self.expandHierarchicalName(list(elog.columns), name)]

            self.results['sums'] = dict(elog.sum())
            #print(f"keys of sums: {self.results['sums']}")

            self.results['n_instructions'] = self.results['sums'][
                self.expandHierarchicalName(self.results['sums'].keys(),
                                            'CPU n instructions')]

            # Parse static power log
            paths = [
                self.testdir / 'powerModelChannel_static_power_log.csv',
                self.testdir / 'powerModelChannel_static_power_log.csv.gz'
            ]

            if not np.any([p.exists() for p in paths]):
                raise Exception(f"Static power log missing from {paths}")

            for path in paths:
                try:
                    if path.exists():
                        splog = pd.read_csv(path)
                        break
                except Exception as e:
                    raise Exception(
                        f"{e}\n ...Encountered while reading file {path}")

            timestep = splog['time(s)'][3] - splog['time(s)'][2]
            print(f"Timestep = {timestep} s")

            for m in filter(lambda x: x != 'time(s)', splog.columns):
                self.results[f'{m}.static_energy'] = splog[m].sum() * timestep
                #print(f'{m}.static energy (J) = {splog[m].sum() * timestep}')

            self.dumpSummary()

    def dumpSummary(self):
        with open(self.testdir / 'summary.txt', 'w+') as of:
            for k, v in self.results.items():
                of.write(f'{k}: {v}\n')

    def loadSummary(self):
        if self.completed and not (self.testdir / 'summary.txt').exists():
            self.parseResults
            return
        self.results = yaml.load(open(self.testdir / 'summary.txt',
                                      'r').read(),
                                 Loader=yaml.FullLoader)

    def deleteFile(self, fn: str):
        """ Delete a file within testdir."""
        if Path.exists(self.testdir / fn):
            (self.testdir / fn).unlink()

    def compressFile(self, fn: str):
        """ Compress a file within testdir """
        import gzip
        f = self.testdir / fn
        if Path.exists(f):
            with open(f, 'rb') as f_in:
                with gzip.open('{}.gz'.format(f), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            f.unlink()  # Delete original file

    def deleteTraces(self):
        """ Delete eventlog, vcd, and tab traces to save disk space. """
        if not self.completed:
            return
        self.deleteFile('powerModelChannel_eventlog.csv')
        self.deleteFile('powerModelChannel_statelog.csv')
        self.deleteFile('ext.tab')
        self.deleteFile('ext.vcd')
        self.deleteFile('cpu.vcd')

    def compressTraces(self):
        """ Compress eventlog, vcd, and tab traces to save disk space. """
        if not self.completed:
            return
        self.compressFile('powerModelChannel_eventlog.csv')
        self.compressFile('powerModelChannel_static_power_log.csv')
        self.compressFile('ext.tab')
        self.compressFile('ext.vcd')
        self.compressFile('cpu.vcd')

    def expandHierarchicalName(self, valid_names, partial_name: str):
        nWords = len(partial_name.split('.'))
        foundKeys = []
        for k in valid_names:
            if '.'.join(k.split('.')[-nWords:]) == partial_name:
                foundKeys.append(k)

        if len(foundKeys) == 0:
            raise Exception(
                f'No entries that match Key {partial_name:s} found in {valid_names}'
            )
        elif len(foundKeys) > 1:
            raise Exception(
                f'Key {partial_name:s} is not unique, but matches {foundKeys}')
        return foundKeys[0]

    def sum_group_event_pi(self, group_name: str, value_name: str):
        """
        Get the sum of results of a value over a group of modules, e.g. reads
        for all banks in a memory.
        """

        columns = [
            k for k in self.results['sums'].keys()
            if group_name in k and value_name in k
        ]

        if len(columns) == 0:
            raise Exception(
                f'No entries that match group {group_name:s} and value {value_name:s} found in {self.results.keys()}'
            )

        return np.sum(np.array(
            self.results[k] for k in columns)) / self.results['n_instructions']

    def get_static_epi(self, module_name: str) -> float:
        fullKey = self.expandHierarchicalName(self.results.keys(),
                                              module_name + '.static_energy')
        return self.results[fullKey] / self.results['n_instructions']

    def get_static_group_epi(self, group_name: str) -> float:
        columns = [
            k for k in self.results.keys()
            if group_name in k and '.static_energy' in k
        ]

        if len(columns) == 0:
            raise Exception(
                f'No entries that match group {group_name:s} and .static found in {self.results.keys()}'
            )

        if self.results['n_instructions'] in [0.0, float('nan'), float('inf')]:
            raise Exception(
                f"Invalid n_instructions '{self.results['n_instructions']}'")

        return np.sum([self.results[k]
                       for k in columns]) / self.results['n_instructions']

    def get_epi(self, event_name: str) -> float:
        fullKey = self.expandHierarchicalName(self.results['sums'].keys(),
                                              event_name)
        return self.results['sums'][fullKey] * self.fusedConfig.get(
            fullKey) / self.results['n_instructions']

    def get_group_epi(self,
                      group_name: str,
                      event_name: str,
                      exclude: str = "XXXXXXXXXXXX") -> float:
        columns = [
            k for k in self.results['sums'].keys() if group_name in k
            and k.split(' ')[-1] == event_name and exclude not in k
        ]

        if len(columns) == 0:
            raise Exception(
                f"No entries that match group {group_name:s} and ends with "
                f"event {event_name:s} found in {self.results['sums'].keys()}")
        """
        if group_name == 'invm':
            print(
                f"{self.name}: "
                f"get_group_epi(group_name = '{group_name}',event_name="
                f"'{event_name}' found columns {columns}."
                f"with energy = {self.fusedConfig.get(group_name + ' ' + event_name)}"
            )
            print([self.results['sums'][k] for k in columns])
        """

        return self.fusedConfig.get(group_name + " " + event_name) * np.sum(
            [self.results['sums'][k]
             for k in columns]) / self.results['n_instructions']

    @property
    def nvm_write_pi(self):
        return self.results['invm_write_pi'] + self.results['dnvm_write_pi']

    @property
    def nvm_read_pi(self):
        return self.results['invm_read_pi'] + self.results['dnvm_read_pi']

    @property
    def cache_write_pi(self):
        return self.results['icache_write_pi'] + self.results['dcache_write_pi']

    @property
    def cache_read_pi(self):
        return self.results['icache_read_pi'] + self.results['dcache_read_pi']

    @property
    def icache_read_epi(self) -> float:
        res = self.instruction_tag_epi + self.results[
            'icache_read_pi'] * self.fusedConfig.get('icache bytes read')
        return res

    @property
    def icache_write_epi(self) -> float:
        res = self.instruction_tag_epi + self.results[
            'icache_write_pi'] * self.fusedConfig.get('icache bytes written')
        return res

    @property
    def dcache_read_epi(self) -> float:
        res = self.data_tag_epi + self.results[
            'dcache_read_pi'] * self.fusedConfig.get('dcache bytes read')
        return res

    @property
    def dcache_write_epi(self) -> float:
        res = self.data_tag_epi + self.results[
            'dcache_write_pi'] * self.fusedConfig.get('dcache bytes written')
        return res

    @property
    def invm_write_epi(self):
        return self.results['invm_write_pi'] * self.fusedConfig.get(
            'invm bytes written')

    @property
    def dnvm_write_epi(self):
        return self.results['dnvm_write_pi'] * self.fusedConfig.get(
            'dnvm bytes written')

    @property
    def invm_read_epi(self):
        return self.results['invm_read_pi'] * self.fusedConfig.get(
            'invm bytes read')

    @property
    def dnvm_read_epi(self):
        return self.results['dnvm_read_pi'] * self.fusedConfig.get(
            'dnvm bytes read')

    @property
    def isram_write_epi(self):
        return self.results['isram_write_pi'] * self.fusedConfig.get(
            'isram bytes written')

    @property
    def dsram_write_epi(self):
        return self.results['dsram_write_pi'] * self.fusedConfig.get(
            'dsram bytes written')

    @property
    def isram_read_epi(self):
        return self.results['isram_read_pi'] * self.fusedConfig.get(
            'isram bytes read')

    @property
    def dsram_read_epi(self):
        return self.results['dsram_read_pi'] * self.fusedConfig.get(
            'dsram bytes read')

    @property
    def dcache_read_epi(self) -> float:
        # nreads * (lineWidth + tagBits * nsets)
        res = self.instruction_tag_epi + self.results[
            'dcache_read_pi'] * self.fusedConfig.get('dcache bytes read')
        return res

    @property
    def instruction_epi(self):
        return self.isram_read_epi + self.isram_write_epi + \
                self.icache_read_epi + self.icache_write_epi + \
                self.invm_read_epi + self.invm_write_epi

    @property
    def data_epi(self):
        return (self.results['data_read_epi'] +
                self.results['data_write_epi'] + self.results['data_tag_epi'])

    @property
    def data_read_epi(self):
        return (self.results['dcache_read_pi'] * E_CACHE_READ +
                self.results['dnvm_read_pi'] * E_MRAM_READ)

    @property
    def data_write_epi(self):
        return (self.results['dcache_write_pi'] * E_CACHE_WRITE +
                self.results['dnvm_write_pi'] * E_MRAM_WRITE)

    @property
    def data_tag_epi(self):
        return (self.results['dcache_tag_access_bits_pi'] *
                self.fusedConfig.get('dcache tag access bits'))

    @property
    def instruction_tag_epi(self):
        return (self.results['icache_tag_access_bits_pi'] *
                self.fusedConfig.get('icache tag access bits'))

    @property
    def completed(self):
        return Path.exists(self.testdir / 'done')

    @property
    def parsed(self):
        return Path.exists(self.testdir / 'summary.txt')

    @property
    def failed(self):
        return Path.exists(self.testdir / 'failed')

    def setFailed(self):
        (self.testdir / Path('failed')).touch()

    @property
    def dataCacheEnabled(self):
        return self.fusedConfig.get('dcache.Enable')

    @property
    def instructionCacheEnabled(self):
        return self.fusedConfig.get('icache.Enable')

    @property
    def dataCacheConfig(self):
        if not self.dataCacheEnabled:
            return (0, 0, 0)
        return (int(self.fusedConfig.get('dcache.CacheLineWidth')),
                int(self.fusedConfig.get('dcache.CacheNLines')),
                int(self.fusedConfig.get('dcache.CacheNSets')))

    @property
    def instructionCacheConfig(self):
        if not self.instructionCacheEnabled:
            return (0, 0, 0)
        return (int(self.fusedConfig.get('icache.CacheLineWidth')),
                int(self.fusedConfig.get('icache.CacheNLines')),
                int(self.fusedConfig.get('icache.CacheNSets')))

    @property
    def instructionCacheTagBits(self):
        if not self.instructionCacheEnabled:
            return 0
        offsetBits = int(math.log2(self.instructionCacheConfig[0]))
        lines = self.instructionCacheConfig[1] * self.instructionCacheConfig[2]
        indexBits = int(math.log2(self.instructionCacheConfig[2]))
        return lines * (int(math.log2(32 * 1024)) - offsetBits - indexBits)

    @property
    def dataCacheTagBits(self):
        if not self.dataCacheEnabled:
            return 0
        offsetBits = int(math.log2(self.dataCacheConfig[0]))
        lines = self.dataCacheConfig[1] * self.dataCacheConfig[2]
        indexBits = int(math.log2(self.dataCacheConfig[2]))
        return lines * (int(math.log2(32 * 1024)) - offsetBits - indexBits)

    @property
    def instructionCacheSizeBits(self):
        res = 1
        for v in self.instructionCacheConfig:
            res *= v
        return 8 * res + self.instructionCacheTagBits

    @property
    def dataCacheSizeBits(self):
        res = 1
        for v in self.dataCacheConfig:
            res *= v
        return 8 * res + self.dataCacheTagBits

    @property
    def start_time(self) -> float:
        if not self.completed:
            return float('Inf')

        with open(self.testdir / Path('stdout.log'), 'r') as infile:
            content = infile.read()

        return 1e-9 * float(
            re.search(r'mcu\.mon: (?P<start>[0-9\.]+) ns 0x00000001',
                      content)['start'])

    @property
    def end_time(self) -> float:
        if not self.completed or self.checkLogsForFails():
            return float('Inf')

        with open(self.testdir / Path('stdout.log'), 'r') as infile:
            content = infile.read()

        endPatterns = [
            r"mcu\.mon: (?P<end>[0-9\.]+) ns 0x00000d1e",
            r"ioSimulationStopper: Simulation stopping at (?P<end>[0-9\.]+) ns after IO count: \d+"
        ]
        for p in endPatterns:
            end = re.search(p, content)
            if end is not None:
                return 1e-9 * float(end['end'])

        if end is None:
            print(f'{self.name}: Failed to parse end time, returning Inf')
            print(f"... in {self.testdir / 'stdout.log'}")

            return float('Inf')

    @property
    def runtime(self) -> float:
        if not self.completed:
            return float('Inf')
        content = ''
        with open(self.testdir / Path('stdout.log'), 'r') as infile:
            content = infile.read()
        try:
            return (self.end_time - self.start_time)
        except Exception as e:
            print(f'{self.name}: Failed to parse runtime')
            print(e)
            print(f"... in {self.testdir / 'stdout.log'}")
            return float('Inf')

    @property
    def completion_times(self) -> Sequence[float]:
        """
        Return a list of completion times, measured as the time between
        posedges of pin 1.
        """
        if not self.completed:
            print(
                f"Experiment::completion_times: Warning: {self.name} has not completed, returning [Inf]"
            )
            return np.array([float('Inf')])
        if self.checkLogsForFails():
            print(
                f"Experiment::completion_times: Warning: {self.name}-{self.hash:08x} has failed, returning [Inf]"
            )
            return np.array([float('Inf')])

        with open(self.testdir / Path('stdout.log'), 'r') as infile:
            content = infile.read()
        try:
            start = 1e-9 * np.array(re.findall(
                r"ioSimulationStopper: @([0-9\.]+) ns trigger count", content),
                                    dtype='float64')

            if len(start) <= 1:
                print(
                    f"Experiment::completion_times: Warning: {self.name} has not completed any iterations, returning [Inf]"
                )
                return np.array([float('Inf')])

            return start[1:] - start[:-1]
        except Exception as e:
            print(f'{self.name}: Failed to parse runtime')
            print(e)
            print(f"... in {self.testdir / 'stdout.log'}")
            return np.array([float('Inf')])

    @property
    def ontime(self) -> float:
        """
        Calculate approximate total on-time as the time keep-alive is asserted
        """
        if not self.completed:
            return float('Inf')

        with open(self.testdir / Path('stdout.log'), 'r') as infile:
            content = infile.read()

        try:
            # find all start / end times and convert  to np array
            start = 1e-9 * np.array(re.findall(
                r'gpio0: @([0-9\.]+) ns posedge on pin 5', content),
                                    dtype='float64')
            end = 1e-9 * np.array(re.findall(
                r'gpio0: @([0-9\.]+) ns negedge on pin 5', content),
                                  dtype='float64')

            # Calculate sum of differences ( each end time - start time)
            # Start is usually 1 longer than end, because the last power cycle
            # stops in the middle
            ontimes = end - start[0:len(end)]
            ontime = np.sum(ontimes)

            # Add last power cycle
            try:
                simEnd = self.end_time
            except Exception as e:
                # App didn't complete
                return float('Inf')

            if len(start) > len(end):
                ontime += simEnd - start[-1]

            return ontime
        except Exception as e:
            print('Failed to parse ontime')
            print(e)
            print(f"... in {self.testdir / 'stdout.log'}")
            return float('Inf')

    def maxEnduranceWritesPerBank(self, memory: str):
        """
        Returns the maximum number of writes to the same location for each memory bank that has writes.
        Returns a list of tuples of (bankNumber, numberOfWrites, addressOfWrites)
        """
        if not self.completed:
            return [float('Inf')]
        content = ''
        with open(self.testdir / Path('stdout.log'), 'r') as infile:
            content = infile.read()

        ex = memory + r".bank_(?P<bank>\d+): max (?P<writes>\d+) writes to 0x(?P<address>[0-9a-f]+)"
        try:
            return re.findall(ex, content)
        except Exception as e:
            print(e)
            print('... in {}'.format((self.testdir / Path('stdout.log')).name))
            return [(np.inf, 0, 0)]

    def maxEnduranceWrites(self, memory: str) -> int:
        """
        Returns the maximum number of writes to the same location for all banks
        in a memory.
        """
        return max(n for _, n, _ in self.maxEnduranceWritesPerBank(memory))

    @property
    def suspendVoltages(self):
        if not self.completed:
            return [float('Inf')]
        content = ''
        with open(self.testdir / Path('stdout.log'), 'r') as infile:
            content = infile.read()
        try:
            vcc = np.array([
                float(v) for v in re.findall(
                    r"svs\.vdet: @\d+ ns output turned off at v_cap=(\d+\.\d+)",
                    content)
            ])
            return vcc if len(vcc) > 0 else float('Inf')
        except Exception as e:
            print(e)
            print('... in {}'.format((self.testdir / Path('stdout.log')).name))
            return np.array([float('Inf')])

    @property
    def meanSuspendVoltage(self):
        return np.mean(self.suspendVoltages)

    @property
    def suspendTimes(self):
        if not self.completed:
            return [float('Inf')]
        content = ''
        with open(self.testdir / Path('stdout.log'), 'r') as infile:
            content = infile.read()
        try:
            start = 1e-9 * np.array(re.findall(
                r'@(\d+) ns handling exception with ID 16', content),
                                    dtype='float64')
            end = 1e-9 * np.array(re.findall(
                r"svs\.vdet: @(\d+) ns output turned off at v_cap", content),
                                  dtype='float64')
            if len(end) != len(start):
                print(start)
                print(
                    f"Experiment::suspendTimes end={len(end)} start={len(start)}"
                )
            return end - start[:len(end)]
        except Exception as e:
            print(e)
            print('... in {}'.format((self.testdir / Path('stdout.log')).name))
            return np.array([float('Inf')])

    @property
    def suspendEnergies(self):
        if not self.completed:
            print(
                f"Experiment::suspendEnergies warning {self.name} has not completed"
            )
            return [float('Inf')]
        # Calculate suspend energy based on voltage drop and time
        vwarn = self.fusedConfig.get("VoltageWarning")
        cap = self.fusedConfig.get("CapacitorValue")
        supply = self.fusedConfig.get("PowerSupplyPower")
        vdropEnergy = np.array(
            [0.5 * cap * ((vwarn**2) - (v**2)) for v in self.suspendVoltages])
        supplyEnergy = self.suspendTimes * supply
        #print(f"vdropEnergy={vdropEnergy*1e9},\nsupplyEnergy={supplyEnergy*1e9}")
        return vdropEnergy + supplyEnergy

    @property
    def nPowerCycles(self):
        if not self.completed:
            return float('Inf')
        content = ''
        with open(self.testdir / Path('stdout.log'), 'r') as infile:
            content = infile.read()
        try:
            return len(re.findall('power on', content))
        except Exception as e:
            print(e)
            print(content)
            return float('NaN')

    @property
    def averageOnTime(self):
        if not self.completed:
            return float('NaN')
        content = ''
        with open(self.testdir / Path('stdout.log'), 'r') as infile:
            content = infile.read()
        try:
            on = np.array(re.findall(r'(\d+.\d+) ns power on', content),
                          dtype='float')
            off = np.array(re.findall(r'(\d+.\d+) ns power off', content),
                           dtype='float')
            period = off[1:] - on[:len(off) - 1]
            return np.mean(period) * 1e-9
        except Exception as e:
            print(e)
            print(content)
            return float('NaN')

    def elfsize(self, section):
        cmd = r'arm-none-eabi-size -A -d {} | grep -oP "\.{}\s+\K(\d+)"'.format(
            self.programElfPath, section)
        try:
            res = sp.run(cmd, shell=True, check=True, stdout=sp.PIPE)
            return int(res.stdout)
        except Exception as e:
            print(e)
            return 0

    @property
    def codesize(self):
        return self.elfsize('text') + self.elfsize('vectors')
        # + self.elfsize( 'rodata') # embedded in .text

    @property
    def datasize(self):
        return self.elfsize('bssbackup') + self.elfsize('data') + self.elfsize(
            'mmdata') + self.elfsize('stack') + self.elfsize('heap')


if __name__ == "__main__":
    """ Set up and run a dummy test """
    fc = FusedConfig()
    fc.set('GdbServer', 'False')
    fc.set('EventLogStart', 'event')
    e = Experiment(odir='/tmp/test',
                   programPath='/tmp/',
                   programName='aes',
                   fusedBinaryPath='${FUSED_ROOT}/build/fused',
                   fusedConfig=fc)
    e.run()
