#/usr/bin/env/python
# encoding: utf-8

'''
Exhaustively search for the ground state electron configuration of the given DB 
layout.
'''

__author__      = 'Samuel Ng'
__copyright_    = 'Apache License 2.0'

import signal
import numpy as np
from scipy.spatial import distance
import itertools
import multiprocessing as mp
import time
from collections import namedtuple
from types import SimpleNamespace

from siqadtools import siqadconn


dp = 6
zero_diff = 10**(-dp)
equal = lambda a, b: abs(a - b) < zero_diff
less_than = lambda a, b: (b - a) > zero_diff
greater_than = lambda a, b: (a - b) > zero_diff
ffmt = lambda x: f'{x:.{dp}f}'


ChargeConfig = namedtuple('ChargeConfig', ['config', 'energy', 'validity'])

class GracefulExit(Exception):
    pass

def sigterm_handler(signum, frame):
    raise GracefulExit('Gracefully exiting.')

class ExhaustiveGroundStateSearcher:
    '''Exhaustively find the ground state configuration of the given DB layout.
    '''

    result_configs = []

    # lattice vector
    lv = SimpleNamespace()
    lv.a = 384e-12
    lv.b = 768e-12
    lv.c = 225e-12

    # physical constants
    pc = SimpleNamespace()
    pc.q0 = 1.602e-19      # elementary charge (C)
    pc.k_b = 8.617e-5      # Boltzmann constant (eV/K)
    pc.eps0 = 8.854e-12    # vacuum permittivity (F/m)
    pc.mudiff = 0.59       # potential between mu_- and mu_+ (eV)

    def __init__(self, dbs, muzm=-0.25, debye=5., epsilon_r=5.6, v_ext=[], 
            verbose=False):
        '''
        Initialize Exhaustive Ground State Searcher.

        Args:
            dbs:    List of DBs in lattice coordinates.
            muzm:   mu_- in SiQAD paper (eV).
            debye:  Thomas-Fermi screening length (nm).
            epsilon_r:  Relative permittivity.
            v_ext:  External potential at each DB (V).
            verbose:        Print lots.
        '''
        self.phys = SimpleNamespace()
        self.runc = SimpleNamespace()

        # store and compute needed vars
        self.phys.dbs = dbs
        self.phys.muzm = muzm
        self.phys.mupz = muzm - self.pc.mudiff
        self.phys.v_ext = np.asarray(v_ext) if len(v_ext) != 0 else np.zeros(len(dbs))

        self.runc.verbose = verbose

        # precalculations
        debye *= 1e-9
        K_c = 1./(4 * np.pi * epsilon_r * self.pc.eps0)
        lat_coord_to_eucl = lambda n, m, l: (n*self.lv.a, m*self.lv.b + l*self.lv.c)
        dbs_eucl = np.asarray([lat_coord_to_eucl(*db) for db in dbs])
        db_r = distance.cdist(dbs_eucl, dbs_eucl, 'euclidean')
        self.neighbor_rank = np.delete(np.argsort(db_r), 0, 1) # first column is always self index
        self.phys.v_ij = np.divide(self.pc.q0 * K_c * np.exp(-db_r/debye), 
                db_r, out=np.zeros_like(db_r), where=db_r!=0)
        if self.runc.verbose:
            print(f'v_ij=\n{self.phys.v_ij}')

    def ground_state_search(self, base=3, num_threads=-1, result_scope='ground'):
        '''
        Search for the ground state using multiple threads.

        Args:
            base:           State count - 2 for {DB-, DB0}, 3 for 
                            {DB-, DB0, DB+}.
            num_threads:    Number of threads for searching, -1 for using all 
                            available CPUs.
            result_scope:   Scope of result to include: 'ground' for only 
                            ground state, 'valid' for all metastable states,
                            'all' for all states. Beware of 'all' as that's
                            len(db)^base total results to write-out.
        
        Returns:
            A list of ChargeConfig objects. The types of configs returned 
            depends on the setting result_scope.
        '''

        max_config_id = base**len(self.phys.dbs)
        self.phys.base = base
        self.runc.result_scope = result_scope

        # managed lists for safe multithreaded access
        manager = mp.Manager()
        managed_result_configs = manager.list([])
        managed_cpu_time_list = manager.list([])

        # prepare threads
        if num_threads <= 0 or num_threads > max_config_id:
            num_threads = min(mp.cpu_count(), max_config_id)
        configs_per_thread = int(np.ceil(max_config_id / num_threads))
        curr_range = (0, configs_per_thread)
        threads = []
        processes = []
        thread_id = 0

        while curr_range[1] <= max_config_id and curr_range[0] != curr_range[1]:
            th = _SearchThread(curr_range, managed_result_configs, 
                    managed_cpu_time_list, thread_id, self.phys, self.runc)
            p = mp.Process(target=th.run)
            threads.append(th)
            processes.append(p)
            curr_range = curr_range[1], min(curr_range[1]+configs_per_thread, max_config_id)
            thread_id += 1

        #wall_time_start = time.time()

        try:
            [p.start() for p in processes]
            [p.join() for p in processes]
        except GracefulExit:
            [p.terminate() for p in processes]

        #self.wall_time_elapsed = time.time() - wall_time_start

        # find the actual ground states among the returned states
        gs_energy = float('inf')
        for charge_config in managed_result_configs:
            if self.runc.verbose:
                print(f'Config: {charge_config.config}, E: {charge_config.energy}')
            if result_scope != 'ground' or equal(charge_config.energy, gs_energy):
                self.result_configs.append(charge_config)
            elif result_scope == 'ground' and charge_config.energy < gs_energy:
                self.result_configs.clear()
                self.result_configs.append(charge_config)
                gs_energy = charge_config.energy
        #self.cpu_time = np.sum(managed_cpu_time_list)

        return self.result_configs


    def check_specified_configs(self, configs):
        '''
        Check the metastability of the provided configs.

        Returns:
            A list of charge configs found to be metastable within the input 
            configs list. If none satisfy metastability conditions, return an 
            empty list.
        '''

        config_iter = ChargeConfigIter(self.base, self.dbs, self.mu, self.v_ext,
                self.epsilon_r, self.debye_length, config_list=accepted_gs, 
                verbose=True)

        has_valid_config = False
        has_next = True
        while has_next:
            has_valid_config = has_valid_config or config.physically_valid()
            has_next = config.advance()

        print(f'Found ground state in provided list: {has_valid_config}')
        return has_valid_config


class _SearchThread:
    '''A single search thread.'''

    def __init__(self, search_range, managed_result_configs, managed_time_list, 
            tid, phys, runc):
        '''
        Args:
            search_range:           Search config index range.
            managed_result_configs: Managed list for writing output configs to.
            managed_time_list:      Managed list for writing time stats to.
            tid:    Thread ID.
            phys:   Input physical parameters.
            runc:   Runtime parameters.
        '''
        self.managed_result_configs = managed_result_configs
        self.managed_time_list = managed_time_list
        self.thread_id = tid
        self.result_scope = runc.result_scope
        self.verbose = runc.verbose

        self.config_iter = _ChargeConfigIter(phys, start_ind=search_range[0], 
                end_ind=search_range[1], verbose=runc.verbose)

    def run(self):
        time_start = time.process_time()
        gs_configs = []
        gs_energy = float('inf')
        result_configs = []

        has_next = True
        while has_next:
            valid = self.config_iter.physically_valid()
            if self.result_scope == 'all' or (valid and self.result_scope == 'valid'):
                energy = self.config_iter.system_energy()
                result_configs.append(ChargeConfig(self.config_iter.n.copy(), energy, int(valid)))
            elif valid:
                energy = self.config_iter.system_energy()
                if less_than(energy, gs_energy):
                    gs_configs.clear()
                    gs_configs.append(self.config_iter.n.copy())
                    gs_energy = energy
                elif equal(energy, gs_energy):
                    gs_configs.append(self.config_iter.n.copy())
            has_next = self.config_iter.advance()

        if self.verbose:
            print(f'Found ground states: {gs_configs}')

        for gs_config in gs_configs:
            result_configs.append(ChargeConfig(gs_config, gs_energy, 1))
        self.managed_result_configs.extend(result_configs)
        self.managed_time_list.append(time.process_time()-time_start)


class _ChargeConfigIter:
    '''
    Registers representing a charge configuration, can be 2 or 3 states.
    '''

    def __init__(self, phys, start_ind=-1, end_ind=-1, only_these_configs=[], 
            verbose=False):
        '''
        Args:
            phys:       Input physical parameters.
            start_ind:  Offset from beginning if `only_these_configs` is empty.
            end_ind:    Ending index (exclusive).
            only_these_configs: Ignore start and end inds, check only these 
                                specific configs.
        '''
        if not only_these_configs and (start_ind < 0 or start_ind >= end_ind):
            raise ValueError('Either only_these_configs, or a valid start_ind, '
                    'end_ind combination must be provided.')
        if phys.base not in [2,3]:
            raise ValueError('Only support 2 or 3 state computation, '
                    f'{phys.base} requested.')

        self.phys = phys
        self.state_max = phys.base - 2
        self.n = np.full(len(phys.dbs), -1)  # -1, 0, +1 for DB-, DB0, DB+

        self.only_these_configs = only_these_configs
        if only_these_configs:
            self.curr_ind = 0
            self.end_ind = len(self.only_these_configs)
            self.n = self.only_these_configs[0]
        else:
            if start_ind != 0:
                i = start_ind
                db_ind = len(phys.dbs) - 1
                while i:
                    i, r = divmod(i, phys.base)
                    if r == 1:
                        self.n[db_ind] = 0
                    elif r == 2:
                        self.n[db_ind] = 1
                    db_ind -= 1

            self.curr_ind = start_ind
            self.end_ind = min(end_ind, phys.base**len(phys.dbs))

        # local potentials
        self.v_i = np.full(len(phys.dbs), -float('inf'))    # Overwritten over each advance
        self.v_i_ready = False

        self.verbose = verbose

    def advance(self):
        '''
        Advance the configuration into the next state.

        Returns:
            A bool informing the success state of the advancement. False when 
            the config has reached the end.
        '''

        self.curr_ind += 1
        if self.curr_ind >= self.end_ind:
            return False

        if self.only_these_configs:
            self.n = self.only_these_configs[self.curr_ind]
        else:
            charge_ind = len(self.n) - 1
            carry = 1
            while charge_ind >= 0 and carry > 0:
                if self.n[charge_ind] != self.state_max:
                    # increment charge
                    carry = 0
                    self.n[charge_ind] += 1
                else:
                    # reset charge and add carry
                    carry = 1
                    self.n[charge_ind] = -1
                charge_ind -= 1

        self.v_i[:] = -float('inf')
        self.v_i_ready = False

        return True

    def system_energy(self, use_qubo=False):
        '''
        Calculate and return the system energy either in paper form or in QUBO
        form.

        Args:
            use_qubo:   Use the QUBO formulation if set to true.

        Returns:
            The system energy in paper form or in QUBO form in float.
        '''

        # TODO implement QUBO formulation
        
        if not self.v_i_ready:
            for i in range(len(self.phys.dbs)):
                if self.v_i[i] == -float('inf'):
                    self._calc_v_i(i)
            self.v_i_ready = True
        #return np.inner(self.v_ext, self.n) + 0.5 * np.inner(self.n, self.v_i)
        return 0.5 * np.inner(self.phys.v_ext, self.n) - 0.5 * np.inner(self.n, self.v_i)
        #return np.inner(self.v_ext, self.n) + 0.5 * np.inner(self.n, np.inner(self.v_ij, self.n))

    def physically_valid(self):
        '''
        Determine the physical validity of the current config. This includes 
        checking the charge config stability and population stability.

        Returns:
            A bool indicating whether this config is physically valid.
        '''

        # population stability and calculate v_i as needed
        for i in range(len(self.phys.dbs)):
            self._calc_v_i(i)
            valid = (self.n[i] == -1 and less_than(self.v_i[i] + self.phys.muzm, 0)) or \
                    (self.n[i] == 1  and greater_than(self.v_i[i] + self.phys.mupz, 0)) or \
                    (self.n[i] == 0  and greater_than(self.v_i[i] + self.phys.muzm, 0) \
                                     and less_than(self.v_i[i] + self.phys.mupz, 0))
            if not valid:
                if self.verbose:
                    print(f'Config {self.n} population unstable, failed at '
                            f'index {i} with v_i={self.v_i[i]}, '
                            f'muzm={self.phys.muzm}, mupz={self.phys.mupz}')
                return False
        self.v_i_ready = True

        # configuration stability
        for i in range(len(self.phys.dbs)):
            # Do nothing with DB+
            if self.n[i] == 1:
                continue

            # Attempt hops from more negative charge states to more positive ones
            for j in range(len(self.phys.dbs)):
                if (self.n[j] > self.n[i]) \
                        and self._hop_energy_delta(i, j) < - zero_diff:
                    if self.verbose:
                        print(f'Config {self.n} charge state '
                                'unstable, failed when hopping from site '
                                f'{i} to {j}')
                    return False
        return True

    def _calc_v_i(self, ind):
        '''
        Calculate the V_i of the given ind and store it in self.v_i[ind].
        '''
        self.v_i[ind] = - self.phys.v_ext[ind] - np.dot(self.phys.v_ij[ind][:], self.n)

    def _hop_energy_delta(self, i, j):
        '''
        Calculate the energy delta from hopping DB at site i to j.

        Returns:
            The energy delta as a float.
        '''
        return - self.v_i[i] + self.v_i[j] - self.phys.v_ij[i][j]


def _siqad_handler(cml_args):
    '''Handle all interactions with SiQAD if invoked from command-line.'''

    print('Setting up problem...')

    # retrieve information from SiQADConnector
    sqconn = siqadconn.SiQADConnector('Exhaustive Ground State Searcher', 
            cml_args.in_file.name, cml_args.out_file)
    dbs = [[db.n, db.m, db.l] for db in sqconn.dbCollection()]

    sq_param = lambda key : sqconn.getParameter(key)

    # check auto_fail threshold and prevent simulation from running if threshold
    # is passed.
    auto_fail_thresh = int(sq_param('auto_fail_threshold'))
    if len(dbs) > int(sq_param('auto_fail_threshold')):
        raise ValueError(f'DB count {len(dbs)} has exceeded the auto fail '
                f'threshold, {auto_fail_thresh}. Please raise the threshold '
                'if you are sure that you want to perform this simulation.')

    # define and retrieve constants
    base = 3 if not cml_args.two_state else 2
    muzm = float(sq_param('global_v0'))
    debye_length = float(sq_param('debye_length'))
    epsilon_r = float(sq_param('epsilon_r'))

    v_ext = np.zeros(len(dbs))
    if cml_args.ext_pots_file != None:
        import json
        with open(cml_args.ext_pots_file, 'r') as f:
            v_ext_load = json.load(f)
            v_ext = v_ext_load['pots'][0]

    # set up runtime configurations if CML or import file have them specified
    num_threads = int(sq_param('num_threads'))
    if cml_args.num_threads != None:
        num_threads = cml_args.num_threads

    result_scope = sq_param('result_scope')
    if cml_args.result_scope and cml_args.result_scope != 'use_input_file':
        result_scope = cml_args.result_scope

    verbose = False
    if cml_args.verbose:
        verbose = True

    egs = ExhaustiveGroundStateSearcher(dbs, muzm=muzm, debye=debye_length, 
            epsilon_r=epsilon_r, v_ext=v_ext, verbose=verbose)

    print('Performing simulation...')
    result_configs = []
    if cml_args.input_configs_validity_check_only:
        specified_configs = _retrieve_specified_configs(cml_args.in_file)
        result_configs = egs.check_specified_configs(specified_configs)
    else:
        result_configs = egs.ground_state_search(base=base, 
                num_threads=num_threads, result_scope=result_scope)

    print('Exporting results...')
    _siqad_export(result_configs, cml_args.export_json, sqconn)


def _retrieve_specified_configs(in_file):
    import lxml.etree as ET
    et = ET.parse(self.in_file)
    accepted_gs = []

    def db_c_to_int(c:str):
        if c == '-':
            return -1
        elif c == '+':
            return 1
        else:
            return 0

    for gs_config_node in et.iter('gs_config'):
        accepted_gs.append([db_c_to_int(c) for c in gs_config_node.text])

    if len(accepted_gs) == 0:
        raise ValueError('There must be more than one ground state config '
                'in the simulation problem template. They should be inside '
                'an <accepted_gs> element with each config enclosed by '
                '<gs_config>.')

    return accepted_gs


def _siqad_export(result_configs, export_json, sqconn):
    # DB locations
    dblocs = []
    for db in sqconn.dbCollection():
        dblocs.append((str(db.x), str(db.y)))
    sqconn.export(db_loc=dblocs)

    # charge configurations
    def config_to_str(config):
        config_str = ''
        for charge in config:
            if charge == 1:
                config_str += '+'
            elif charge == 0:
                config_str += '0'
            elif charge == -1:
                config_str += '-'
            else:
                raise ValueError(f'Unknown charge value: {charge}')
        return config_str
    export_configs = []
    for charge_config in result_configs:
        export_configs.append([config_to_str(charge_config.config), 
            ffmt(charge_config.energy), str(1), str(int(charge_config.validity)), str(3)])
    sqconn.export(db_charge=export_configs)

    if export_json:
        with open(os.path.splitext(sqconn.outputPath())[0]+'.json', 'w') as f:
            json.dump(charge_configs, f)

    # timing information
    #self.sqconn.export(misc=[['time_s_cpu_time', self.cpu_time],
    #    ['time_s_wall_time', self.wall_time_elapsed]])


def parse_cml_args():
    '''Parse command-line arguments.'''
    import argparse

    parser = argparse.ArgumentParser(description='Exhaustively find the ground '
            'state configuration of the given DB layout.')
    parser.add_argument(dest='in_file', type=argparse.FileType('r'), 
            help='Path to the simulation problem file.',
            metavar='IN_FILE')
    parser.add_argument(dest='out_file', help='Path to the '
            'result file.', metavar='OUT_FILE')
    parser.add_argument('--ext-pots-file', dest='ext_pots_file', 
            help='Import external potentials generated by PoisSolver.', 
            metavar='EXT_POTS_FILE')
    parser.add_argument('--num-threads', dest='num_threads', type=int, 
            help='Number of threads to run concurrently, leave blank to use all '
            'threads available.')
    parser.add_argument('--result-scope', dest='result_scope', 
            default='use_input_file', const='ground', nargs='?', 
            choices=['use_input_file', 'ground', 'valid'], help='Indicate '
            'which states to include - ground for only the ground state, valid '
            'for all the valid states, all for everything.')
    parser.add_argument('--input-configs-validation-only', action='store_true', 
            dest='input_configs_validity_check_only', help='Check whether any '
            'of the configs given in accepted_gs->gs_config are ground states. '
            'If the configs are not stable, the script ends without returning '
            'any ground states; if the configs are all stable, the script does '
            'an exhaustive check and returns the ground state that is found. '
            'This mode is intended to speed up parameter sweeping tests.')
    # QUBO has not been fully implemented yet
    #parser.add_argument('--use-qubo-obj-func', action='store_true',
    #        dest='use_qubo_obj_func', help='NOT IMPLEMENTED YET.')
    parser.add_argument('--two-state', action='store_true', dest='two_state')
    parser.add_argument('--export-json', action='store_true', dest='export_json')
    parser.add_argument('--verbose', action='store_true', dest='verbose')
    return parser.parse_args()

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, sigterm_handler)

    print('Performing exhaustive search...')
    _siqad_handler(parse_cml_args())
    print('Finished')
