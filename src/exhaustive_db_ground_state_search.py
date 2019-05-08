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
from tqdm import tqdm
import siqadconn
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

class ExhaustiveGroundStateSearch:
    '''Exhaustively find the ground state configuration of the given DB layout.
    '''

    q0 = 1.602e-19
    eps0 = 8.854e-12
    k_b = 8.617e-5
    elec_configs = []

    lat_a = 3.84    # lattice vector in x, angstroms    (intra dimer row)
    lat_b = 7.68    # lattice vector in y, angstroms    (inter dimer row)
    lat_c = 2.25    # dimer pair separation, angstroms

    def __init__(self, in_file, out_file, verbose=False):
        self.in_file = in_file
        self.out_file = out_file
        self.verbose = verbose

        self.sqconn = siqadconn.SiQADConnector('Exhaustive Ground State Searcher',
                self.in_file.name, self.out_file)

        self.precalculations()

    def precalculations(self):
        '''Retrieve variables from SiQADConnector and precompute variables.'''

        sq_param = lambda key : self.sqconn.getParameter(key)

        self.num_threads = int(sq_param('num_threads'))

        db_scale = 1e-10
        def lat_coord_to_eucl(n, m, l):
            return db_scale * n * self.lat_a, db_scale * (m * self.lat_b + l * self.lat_c)

        self.dbs = [lat_coord_to_eucl(db.n, db.m, db.l) for db in self.sqconn.dbCollection()]
        self.dbs = np.asarray(self.dbs)

        auto_fail_threshold = int(sq_param('auto_fail_threshold'))
        if (auto_fail_threshold > 0 and len(self.dbs) > auto_fail_threshold):
            raise Exception('Input DB count {} is greater than auto fail '
                    'threshold {}. If you are sure that you want to run this '
                    'problem, raise the threshold.'
                    .format(len(self.dbs), auto_fail_threshold))

        # retrieve and process simulation parameters
        K_c = 1./(4 * np.pi * float(sq_param('epsilon_r')) * self.eps0)
        debye_length = float(sq_param('debye_length'))
        debye_length *= 1e-9        # debye_length given in nm

        # precompute distances and inter-DB potentials
        db_r = distance.cdist(self.dbs, self.dbs, 'euclidean')
        self.neighbor_rank = np.delete(np.argsort(db_r), 0, 1) # first column is always self index
        self.v_ij = np.divide(self.q0 * K_c * np.exp(-db_r/debye_length), 
                db_r, out=np.zeros_like(db_r), where=db_r!=0)
        if self.verbose:
            print('v_ij=\n{}'.format(self.v_ij))

        # local potentials
        self.mu = float(sq_param('global_v0'))
        self.mus = np.ones(len(self.dbs)) * self.mu * -1
        self.v_ext = np.zeros(len(self.dbs))  # TODO add support for ext potential
        #self.v_local = np.ones(len(self.dbs)) * -1 * float(sq_param('global_v0'))

    def ground_state_search_mt(self, num_threads, stability_checks='all',
            include_states='ground', use_qubo_obj_func=False):
        '''
        Search for the ground state using multiple threads.

        Args:
            num_threads:        Number of threads to spawn.
            stability_checks:   Options 'population_only' or 'all'.
            include_states:     Options 'ground', 'valid' or 'all'.
            use_qubo_obj_func:  Set to true to use QUBO objective function as 
                                the energy output.
        '''

        max_config_id = 2**len(self.dbs)

        manager = mp.Manager()
        managed_elec_configs = manager.list([])
        managed_cpu_time_list = manager.list([])

        # cml num_threads takes precedent over sqconn param
        if num_threads == None:
            num_threads = self.num_threads
        if num_threads <= 0 or num_threads > max_config_id:
            num_threads = min(mp.cpu_count(), max_config_id)
        
        configs_per_thread = int(np.ceil(max_config_id / num_threads))
        curr_range = (0, configs_per_thread)
        threads = []
        processes = []
        thread_id = 0
        while curr_range[1] <= max_config_id and curr_range[0] != curr_range[1]:
            th = SearchThread(managed_elec_configs, managed_cpu_time_list, 
                    thread_id, curr_range, self.dbs, self.v_ij, self.mu, 
                    stability_checks, include_states, use_qubo_obj_func, 
                    self.verbose)
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
            if less_than(elec_config.energy, gs_energy):
                self.elec_configs.clear()
                self.elec_configs.append(elec_config)
                gs_energy = elec_config.energy
            elif equal(elec_config.energy, gs_energy):
                self.elec_configs.append(elec_config)

        self.cpu_time = np.sum(managed_cpu_time_list)
            
    def export_results(self, export_json=False):
        '''Export the results sorted by the energy given by the objective
        function.'''
        import json

        if export_json:
            with open('result.json', 'w') as outfile:
                #json.dump(sorted(self.elec_configs, key=lambda config: config.energy), outfile)
                json.dump(self.elec_configs, outfile)

        # DB locations
        dblocs = []
        for db in self.sqconn.dbCollection():
            dblocs.append((str(db.x), str(db.y)))
        self.sqconn.export(db_loc=dblocs)

        # charge configurations
        charge_configs = []
        for elec_config in self.elec_configs:
            charge_configs.append([elec_config.config, ffmt(elec_config.energy),
                str(1), str(int(elec_config.validity))])
        self.sqconn.export(db_charge=charge_configs)

        # timing information
        self.sqconn.export(misc=[['time_s_cpu_time', self.cpu_time],
                                 ['time_s_wall_time', self.wall_time_elapsed]])

class SearchThread:
    '''A single search thread.'''


    def __init__(self, managed_config_results, managed_time_list, t_id, 
            search_range, dbs, v_ij, mu, stability_checks, include_states, 
            use_qubo_obj_func, verbose):
        '''search_range is a tuple containing the start and end indices.'''
        self.managed_config_results = managed_config_results
        self.managed_time_list = managed_time_list
        self.thread_id = t_id
        self.search_range = search_range
        self.dbs = dbs
        self.v_ij = v_ij
        self.mu = mu
        self.mus = mu * np.ones(len(dbs))
        self.stability_checks = stability_checks
        self.include_states = include_states
        self.verbose = verbose
        self.use_qubo_obj_func = use_qubo_obj_func

    def run(self):
        time_start = time.process_time()
        elec_configs = []
        gs_configs_str = []
        gs_energy = float('inf')
        for config_id in range(self.search_range[0], self.search_range[1]):
            config_str = np.binary_repr(config_id).zfill(len(self.dbs))
            validity = self.physically_valid(config_str)
            if (self.include_states != 'ground' or validity):
                energy = self.objective_function(config_str) if self.use_qubo_obj_func \
                        else self.system_energy(config_str)

                #if (validity and energy < gs_energy):
                if (validity and less_than(energy, gs_energy)):
                    gs_configs_str.clear()
                    gs_configs_str.append(config_str)
                    gs_energy = energy
                #elif (validity and energy == gs_energy):
                elif (validity and equal(energy, gs_energy)):
                    gs_configs_str.append(config_str)

                if (self.include_states == 'all') or (self.include_states == 'valid' and validity):
                    elec_configs.append(ElectronConfig(
                        config_str,
                        energy,
                        validity
                        ))
        if self.include_states == 'ground':
            for gs_config_str in gs_configs_str:
                elec_configs.append(ElectronConfig(gs_config_str, gs_energy, 1))
        self.managed_config_results.extend(elec_configs)
        self.managed_time_list.append(time.process_time()-time_start)

    def objective_function(self, charge_config):
        '''Return the runtime energy of the given charge configuration 
        treating Fermi level as negative mu and accounting for all Coulombic 
        interactions.'''

        charges = np.asarray([int(c) for c in charge_config])
        return -np.inner(charges, self.mus) + 0.5 * np.inner(charges, np.dot(self.v_ij, charges))

    def system_energy(self, charge_config):
        '''Return the system energy of the given charge configuration 
        accounting for all Coulombic interactions.'''

        charges = np.asarray([int(c) for c in charge_config])
        return .5 * np.inner(charges, np.dot(self.v_ij, charges))

    def physically_valid(self, charge_config):
        '''Return whether the configuration is physically valid.'''
        charges = np.asarray([int(c) for c in charge_config])

        # population stable
        v_local = -np.dot(self.v_ij, charges) # TODO v_ext
        for i in range(len(v_local)):
            if (charges[i] == 1 and less_than(v_local[i] + self.mu, 0)) or \
                    (charges[i] == 0 and greater_than(v_local[i] + self.mu, 0)):
                if self.verbose:
                    print('Config {} population unstable, failed at index {} with v_i={}'
                            .format(charges, i, v_i))
                return False

        # don't need to check configuration stability if not asked to
        if self.stability_checks == 'population_only':
            return True

        # locally minimal
        for i in range(len(charges)):
            if charges[i] != 1:
                continue
            for j in range(len(charges)):
                if i == j or charges[j] != 0:
                    continue
                if less_than(self.hop_energy_delta(charges, v_local, i, j), 0):
                    if self.verbose:
                        print('Charge {} charge state unstable, failed when hopping '
                                'from site {} to {}'.format(charge_config, i,j))
                    return False

        return True

    def hop_energy_delta(self, charges, v_local, i_ind, j_ind):
        '''The energy delta as a result of hopping from site i to j given the 
        charges and v_local lists.'''

        return v_local[i_ind] - v_local[j_ind] - self.v_ij[i_ind][j_ind]

def parse_cml_args():
    '''Parse command-line arguments.'''

    parser = argparse.ArgumentParser(description='Exhaustively find the ground '
            'state configuration of the given DB layout.')
    parser.add_argument(dest='in_file', type=argparse.FileType('r'), 
            help='Path to the simulation problem file.',
            metavar='IN_FILE')
    parser.add_argument(dest='out_file', help='Path to the '
            'result file.', metavar='OUT_FILE')
    parser.add_argument('--num-threads', dest='num_threads', type=int, 
            help='Number of threads to run concurrently, leave blank to use all '
            'threads available.')
    parser.add_argument('--stability-checks', dest='stability_checks',
            default='all', const='all', nargs='?', 
            choices=['population_only', 'all'],
            help='Indicate which stability checks to perform.')
    parser.add_argument('--include-states', dest='include_states', default='ground',
            const='ground', nargs='?', choices=['ground', 'valid', 'all'],
            help='Indicate which states to include - ground for only the ground '
            'state, valid for all the valid states, all for everything.')
    parser.add_argument('--use-qubo-obj-func', action='store_true',
            dest='use_qubo_obj_func')
    parser.add_argument('--export-json', action='store_true', dest='export_json')
    parser.add_argument('--verbose', action='store_true', dest='verbose')
    return parser.parse_args()

if __name__ == '__main__':
    cml_args = parse_cml_args()
    print('Setting up problem...')
    egss = ExhaustiveGroundStateSearch(cml_args.in_file, cml_args.out_file, 
            verbose=cml_args.verbose)
    print('Performing exhaustive search...')
    egss.ground_state_search_mt(cml_args.num_threads, cml_args.stability_checks,
            cml_args.include_states, cml_args.use_qubo_obj_func)
    print('Exporting results...')
    egss.export_results(cml_args.export_json)
    print('Finished')
