#/usr/bin/env/python
# encoding: utf-8

'''
Exhaustively search for the ground state electron configuration of the given DB 
layout.
'''

__author__      = 'Samuel Ng'
__copyright_    = 'Apache License 2.0'
__version__     = '0.1'
__date__        = '2019-05-04'  # last update

from collections import namedtuple
import numpy as np
from scipy.spatial import distance
import argparse
from siqadtools import siqadconn
import itertools
import multiprocessing as mp
import time

ElectronConfig = namedtuple('ElectronConfig', ['config', 'energy', 'validity'])

dp = 6
zero_diff = 10**(-dp)
equal = lambda a, b: abs(a - b) < zero_diff
less_than = lambda a, b: (b - a) > zero_diff
greater_than = lambda a, b: (a - b) > zero_diff
ffmt = lambda x: f'{x:.{dp}f}'


class ChargeConfig:
    '''
    Registers representing a charge configuration, can be 2 or 3 states.
    '''

    # constants
    q0 = 1.602e-19
    eps0 = 8.854e-12
    k_b = 8.617e-5

    def __init__(self, state_count, dbs, mu, v_ext, epsilon_r, debye_length, 
            config_list=[], start_ind=-1, end_ind=-1, verbose=False):
        '''
        Args:
            state_count:    Number of states (2 for DB- and DB0, 3 to also 
                            include DB+).
            dbs:            List of DB locations in Euclidean coordinates in
                            angstrom.
            mu:             Potential between Fermi level and (0/-) charge 
                            transition level.
            epsilon_r:      Relative permittivity.
            debye_length:   Thomas-Fermi Debye length.
            config_list:    List of configs to run, if this is present then 
                            start_ind and end_ind will be ignored.
            start_ind:      Offset from the beginning.
            end_ind:        Ending index (exclusive).
        '''
        if not config_list and (start_ind < 0 or start_ind >= end_ind):
            raise ValueError('Either config_list, or a valid start_ind, end_ind'
                    ' combination must be provided.')
        if state_count not in [2,3]:
            raise ValueError('Only support 2 or 3 state computation, '
                    f'{state_count} requested.')

        self.state_max = state_count - 2
        self.db_states = np.full(len(dbs), -1)          # -1, 0, 1 for DB-, DB0, and DB+

        self.config_list = config_list if config_list else None
        if config_list:
            self.curr_ind = 0
            self.end_ind = len(self.config_list)
            self.db_states = self.config_list[0]
        else:
            if start_ind != 0:
                n = start_ind
                db_ind = len(dbs)-1
                while n:
                    n, r = divmod(n, state_count)
                    if r == 1:
                        self.db_states[db_ind] = 0
                    elif r == 2:
                        self.db_states[db_ind] = 1
                    db_ind -= 1

            # TODO add offset functionality probably through modulus
            self.curr_ind = start_ind
            self.end_ind = min(end_ind, 3**len(dbs))

        self.verbose = verbose

        # retrieve and process simulation parameters
        self.K_c = 1./(4 * np.pi * epsilon_r * self.eps0)
        self.eta = 0.59                         # TODO make configurable
        self.muzm = mu
        self.mupz = mu - self.eta
        self.debye_length = debye_length
        self.v_ext = v_ext

        self.perform_precalculations(dbs)

    def perform_precalculations(self, dbs):
        '''
        Perform precalculations.
        '''

        self.dbs = np.asarray(dbs)

        # precompute distances and inter-DB potentials
        db_r = distance.cdist(self.dbs, self.dbs, 'euclidean')
        self.neighbor_rank = np.delete(np.argsort(db_r), 0, 1) # first column is always self index
        self.v_ij = np.divide(self.q0 * self.K_c * np.exp(-db_r/self.debye_length), 
                db_r, out=np.zeros_like(db_r), where=db_r!=0)
        if self.verbose:
            print('v_ij=\n{}'.format(self.v_ij))

        # local potentials
        self.v_i = np.full(len(self.dbs), -float('inf'))    # Overwritten over each advance
        self.v_i_ready = False

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

        if self.config_list:
            self.db_states = self.config_list[self.curr_ind]
        else:
            charge_ind = len(self.db_states) - 1
            carry = 1
            while charge_ind >= 0 and carry > 0:
                if self.db_states[charge_ind] != self.state_max:
                    # increment charge
                    carry = 0
                    self.db_states[charge_ind] += 1
                else:
                    # reset charge and add carry
                    carry = 1
                    self.db_states[charge_ind] = -1
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
            for i in range(len(self.dbs)):
                if self.v_i[i] == -float('inf'):
                    self._calc_v_i(i)
            self.v_i_ready = True
        #return np.inner(self.v_ext, self.db_states) + 0.5 * np.inner(self.db_states, self.v_i)
        return np.inner(self.v_ext, self.db_states) - 0.5 * np.inner(self.db_states, self.v_i)
        #return np.inner(self.v_ext, self.db_states) + 0.5 * np.inner(self.db_states, np.inner(self.v_ij, self.db_states))

    def physically_valid(self):
        '''
        Determine the physical validity of the current config. This includes 
        checking the charge config stability and population stability.

        Returns:
            A bool indicating whether this config is physically valid.
        '''

        # population stability and calculate v_i as needed
        for i in range(len(self.dbs)):
            self._calc_v_i(i)
            valid = (self.db_states[i] == -1 and less_than(self.v_i[i] + self.muzm, 0)) or \
                    (self.db_states[i] == 1  and greater_than(self.v_i[i] + self.mupz, 0)) or \
                    (self.db_states[i] == 0  and greater_than(self.v_i[i] + self.muzm, 0) \
                                             and less_than(self.v_i[i] + self.mupz, 0))
            if not valid:
                if self.verbose:
                    print(f'Config {self.db_states} population unstable, failed at '
                            f'index {i} with v_i={self.v_i[i]}, muzm={self.muzm}, '
                            f'eta={self.eta}')
                return False
        self.v_i_ready = True

        # configuration stability
        for i in range(len(self.dbs)):
            # Do nothing with DB+
            if self.db_states[i] == 1:
                continue

            # Attempt hops from more negative charge states to more positive ones
            for j in range(len(self.dbs)):
                if (self.db_states[j] > self.db_states[i]) \
                        and self._hop_energy_delta(i, j) < - zero_diff:
                    if self.verbose:
                        print(f'Config {self.db_states} charge state '
                                f'unstable, failed when hopping from site '
                                f'{i} to {j}')
                    return False
        return True

    def _calc_v_i(self, ind):
        '''
        Calculate the V_i of the given ind and store it in self.v_i[ind].
        '''
        self.v_i[ind] = - self.v_ext[ind] - np.dot(self.v_ij[ind][:], self.db_states)

    def _hop_energy_delta(self, i, j):
        '''
        Calculate the energy delta from hopping DB at site i to j.

        Returns:
            The energy delta as a float.
        '''
        return - self.v_i[i] + self.v_i[j] - self.v_ij[i][j]


class ExhaustiveGroundStateSearch:
    '''Exhaustively find the ground state configuration of the given DB layout.
    '''

    elec_configs = []

    lat_a = 3.84    # lattice vector in x, angstroms    (intra dimer row)
    lat_b = 7.68    # lattice vector in y, angstroms    (inter dimer row)
    lat_c = 2.25    # dimer pair separation, angstroms

    def __init__(self, in_file, out_file, ext_pots_file, verbose=False):
        self.in_file = in_file
        self.out_file = out_file
        self.ext_pots_file = ext_pots_file
        self.verbose = verbose

        self.sqconn = siqadconn.SiQADConnector('Exhaustive Ground State Searcher',
                self.in_file.name, self.out_file)

    def ground_state_search_3_states(self, num_threads, stability_checks='all',
            include_states='ground', check_config=False, use_qubo_obj_func=False, 
            two_state=False):
        '''
        Search for the ground state using multiple threads.

        Args:
            num_threads:        Number of threads to spawn.
            stability_checks:   Options 'population_only' or 'all'.
            include_states:     Options 'ground', 'valid' or 'all'.
            check_config:       See argument help for --check-config.
            use_qubo_obj_func:  Set to true to use QUBO objective function as 
                                the energy output.
            two_state:          Run in two-state mode.
        '''

        db_scale = 1e-10    # ang to m
        sq_param = lambda key : self.sqconn.getParameter(key)
        lat_coord_to_eucl = lambda n, m, l: (db_scale * n * self.lat_a, db_scale * (m * self.lat_b + l * self.lat_c))

        if include_states == 'use_input_file':
            include_all_valid = sq_param('include_states') == 'valid'
        else:
            include_all_valid = include_states == 'valid'

        manager = mp.Manager()
        managed_elec_configs = manager.list([])
        managed_cpu_time_list = manager.list([])

        # retrieve sim info
        self.dbs = np.asarray([lat_coord_to_eucl(db.n, db.m, db.l) \
                for db in self.sqconn.dbCollection()])
        self.epsilon_r = float(sq_param('epsilon_r'))
        self.debye_length = float(sq_param('debye_length')) * 1e-9
        self.mu = float(sq_param('global_v0'))
        self.v_ext = np.zeros(len(self.dbs))    # array of zeros by default
        if self.ext_pots_file:
            import json
            with open(self.ext_pots_file, 'r') as f:
                v_ext_load = json.load(f)       # load from JSON if exists
                self.v_ext = v_ext_load['pots'][0]

        self.base = 3 if not two_state else 2

        run_egs = self.check_config_validity() if check_config else True

        if run_egs:
            self.time_keeping = True
            max_config_id = self.base**len(self.dbs)
            # prepare threads
            if num_threads == None:
                num_threads = int(sq_param('num_threads'))
            if num_threads <= 0 or num_threads > max_config_id:
                num_threads = min(mp.cpu_count(), max_config_id)
            configs_per_thread = int(np.ceil(max_config_id / num_threads))
            curr_range = (0, configs_per_thread)
            threads = []
            processes = []
            thread_id = 0
            while curr_range[1] <= max_config_id and curr_range[0] != curr_range[1]:
                th = SearchThreadThreeStates(managed_elec_configs, managed_cpu_time_list, 
                        thread_id, curr_range, self.dbs, self.mu, self.v_ext, self.epsilon_r, 
                        self.debye_length, use_qubo_obj_func, self.base, include_all_valid, self.verbose)
                p = mp.Process(target=th.run)
                threads.append(th)
                processes.append(p)
                curr_range = curr_range[1], min(curr_range[1]+configs_per_thread, max_config_id)
                thread_id += 1

            wall_time_start = time.time()

            [p.start() for p in processes]
            [p.join() for p in processes]

            self.wall_time_elapsed = time.time() - wall_time_start

            # find the actual ground states among the returned states
            gs_energy = float('inf')
            for elec_config in managed_elec_configs:
                if not include_all_valid and less_than(elec_config.energy, gs_energy):
                    self.elec_configs.clear()
                    self.elec_configs.append(elec_config)
                    gs_energy = elec_config.energy
                elif include_all_valid or equal(elec_config.energy, gs_energy):
                    self.elec_configs.append(elec_config)
            self.cpu_time = np.sum(managed_cpu_time_list)
        else:
            self.time_keeping = False


    def check_config_validity(self):
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

        config = ChargeConfig(self.base, self.dbs, self.mu, self.v_ext,
                self.epsilon_r, self.debye_length, config_list=accepted_gs, 
                verbose=True)

        has_valid_config = False
        has_next = True
        while has_next:
            has_valid_config = has_valid_config or config.physically_valid()
            has_next = config.advance()

        print(f'Found ground state in provided list: {has_valid_config}')
        return has_valid_config

    def export_results(self, export_json=False):
        '''Export the results sorted by the energy given by the objective
        function.'''
        import json
        import os

        # DB locations
        dblocs = []
        for db in self.sqconn.dbCollection():
            dblocs.append((str(db.x), str(db.y)))
        self.sqconn.export(db_loc=dblocs)

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
        charge_configs = []
        for elec_config in self.elec_configs:
            charge_configs.append([config_to_str(elec_config.config), 
                ffmt(elec_config.energy), str(1), str(int(elec_config.validity)), str(3)])
        self.sqconn.export(db_charge=charge_configs)

        if export_json:
            with open(os.path.splitext(self.out_file)[0]+'.json', 'w') as outfile:
                json.dump(charge_configs, outfile)

        # timing information
        if self.time_keeping:
            self.sqconn.export(misc=[['time_s_cpu_time', self.cpu_time],
                ['time_s_wall_time', self.wall_time_elapsed]])


class SearchThreadThreeStates:
    '''A single search thread.'''

    def __init__(self, managed_config_results, managed_time_list, t_id, 
            search_range, dbs, mu, v_ext, epsilon_r, debye_length,
            use_qubo_obj_func, states, include_all_valid, verbose):
        '''search_range is a tuple containing the start and end indices.'''
        self.managed_config_results = managed_config_results
        self.managed_time_list = managed_time_list
        self.thread_id = t_id
        self.dbs = dbs
        self.states = states
        self.include_all_valid = include_all_valid
        self.verbose = verbose
        self.use_qubo_obj_func = use_qubo_obj_func

        self.config = ChargeConfig(states, self.dbs, mu, v_ext, epsilon_r, 
                debye_length, start_ind=search_range[0], end_ind=search_range[1],
                verbose=verbose)

    def run(self):
        time_start = time.process_time()
        all_configs = []
        gs_configs = []
        gs_energy = float('inf')
        elec_configs = []

        has_next = True
        while has_next:
            valid = self.config.physically_valid()
            if valid:
                energy = self.config.system_energy()
                if valid and self.include_all_valid:
                    elec_configs.append(ElectronConfig(self.config.db_states.copy(), energy, 1))
                elif valid and less_than(energy, gs_energy):
                    gs_configs.clear()
                    gs_configs.append(self.config.db_states.copy())
                    gs_energy = energy
                elif valid and equal(energy, gs_energy):
                    gs_configs.append(self.config.db_states.copy())

            has_next = self.config.advance()

        if self.verbose:
            print(f'Found ground states: {gs_configs}')

        for gs_config in gs_configs:
            elec_configs.append(ElectronConfig(gs_config, gs_energy, 1))
        self.managed_config_results.extend(elec_configs)
        self.managed_time_list.append(time.process_time()-time_start)


def parse_cml_args():
    '''Parse command-line arguments.'''

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
    parser.add_argument('--stability-checks', dest='stability_checks',
            default='all', const='all', nargs='?', 
            choices=['population_only', 'all'],
            help='Indicate which stability checks to perform.')
    parser.add_argument('--include-states', dest='include_states', default='use_input_file',
            const='ground', nargs='?', choices=['use_input_file', 'ground', 'valid'],
            help='Indicate which states to include - ground for only the ground '
            'state, valid for all the valid states, all for everything.')
    parser.add_argument('--check-config', action='store_true', 
            dest='check_config', help='Check whether any of the configs given '
            'in accepted_gs->gs_config are ground states. If the configs are '
            'not stable, the script ends without returning any ground states; '
            'if the configs are all stable, the script does an exhaustive check'
            ' and returns the ground state that is found. This mode is intended'
            ' to speed up parameter sweeping tests.')
    # QUBO has not been fully implemented yet
    parser.add_argument('--use-qubo-obj-func', action='store_true',
            dest='use_qubo_obj_func', help='NOT IMPLEMENTED YET.')
    parser.add_argument('--two-state', action='store_true', dest='two_state')
    parser.add_argument('--export-json', action='store_true', dest='export_json')
    parser.add_argument('--verbose', action='store_true', dest='verbose')
    return parser.parse_args()

if __name__ == '__main__':
    cml_args = parse_cml_args()
    print('Setting up problem...')
    egss = ExhaustiveGroundStateSearch(cml_args.in_file, cml_args.out_file, 
            cml_args.ext_pots_file, verbose=cml_args.verbose)
    print('Performing exhaustive search...')
    egss.ground_state_search_3_states(cml_args.num_threads, 
            cml_args.stability_checks, cml_args.include_states, 
            cml_args.check_config, cml_args.use_qubo_obj_func, 
            cml_args.two_state)
    print('Exporting results...')
    egss.export_results(cml_args.export_json)
    print('Finished')
