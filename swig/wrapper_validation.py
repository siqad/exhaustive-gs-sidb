#!/usr/bin/env python3
# encoding: utf-8

'''
Validate that the SWIG wrapper works.
'''

__author__      = 'Samuel Ng'
__copyright__   = 'Apache License 2.0'

import exhaustive_gs as egs

def or_gate(in1, in2, expanded_gs=False):
    phys = {'muzm': -0.28, 'debye': 5., 'eps_r': 5.6}
    db_locs = []
    if in1 == 1:
        db_locs.append([-8,-4,0])
    if in2 == 1:
        db_locs.append([4,-4,0])
    db_locs.extend([
            [-6,-3,0],
            [-4,-2,0],
            [0,-2,0],
            [2,-3,0],
            [-2,0,0],
            [-2,1,1],
            [-2,3,1]])
    gs_configs = []
    if in1 == 0 and in2 == 0:                                                
        gs_configs.append([-1,0,0,-1,-1,0,-1])                               
    elif in1 == 1 and in2 == 1:                                              
        gs_configs.append([-1,-1,0,-1,0,0,0,-1,-1])                          
        gs_configs.append([-1,-1,0,0,-1,0,0,-1,-1])                          
        if expanded_gs:
            gs_configs.append([-1,-1,0,-1,-1,0,0,-1,-1])                         
    elif in1 == 0 and in2 == 1:                                              
        gs_configs.append([-1,-1,0,-1,0,0,-1,-1])                            
    else:                                                                    
        gs_configs.append([-1,0,-1,0,-1,0,-1,-1])
    return (db_locs, phys, gs_configs)

if __name__ == '__main__':
    db_locs, phys, expected_confs = or_gate(0, 0)

    # set up simulation parameters
    sp = egs.SimParams()
    sp.set_db_locs(db_locs)
    sp.set_v_ext([0.0 for i in range(len(db_locs))])
    for k,v in phys.items():
        sp.set_param(k, v)
    sp.print_phys_params()

    # invocation
    egs_eng = egs.EGS(sp)
    egs_eng.invoke()

    # check result
    # NOTE this wouldn't raise an error as long as one of the expected ground 
    # states is reached.
    confs = [result.config for result in egs_eng.gs_results()]
    if not any([conf in expected_confs for conf in confs]):
        print(f'Expected ground state not reached. Retrieved: {confs}; Expected: {expected_confs}')
        raise

    print('Tests ended successfully.')
