__author__ = 'syarkoni'

import utilities
import numpy as np

from SaaSClient import SaaSClient

# url = 'https://dw2x.dwavesys.com/sapi/'
# alpha_token = ''
# token = ''


url = 'https://cloud.dwavesys.com/sapi/'
token = ''


def test():

    params = {'n': 10, 'm': 4}
    prob = utilities.generate_problem('BA', params, 0)
    d_qubo = {(i, j): val for [i, j, val] in prob['Q']}

    solver = utilities.local_connection.get_solver("c4-sw_optimize")
    adj = utilities.get_hardware_adjacency(solver)
    emb = utilities.get_embedding(prob['_id'], "c4-sw_optimize", seed=0, adj=adj)

    emb_Q = utilities.apply_embedding(d_qubo, emb['chains'], adj, 10)
    params = {"answer_mode": "histogram", "num_reads": 10, 'chain_strength': 10}


    res = utilities.solve_dwave(solver, prob['_id'], emb_Q, emb['_id'], solver_name="c4-sw_optimize", **params)


def first_experiment():
    problem_ids = list()
    result_ids = list()

    rc = utilities.RemoteConnection(url, token)
    solver = rc.get_solver('BAY1_2000Q')
    # solver = utilities.local_connection.get_solver("c4-sw_optimize")
    adj = utilities.get_hardware_adjacency(solver)

    for i in range(10):
        params = {'n': 50, 'm': 20}
        print 'Generating problem: ', i+1
        prob = utilities.generate_problem('BA', params, seed=i)
        problem_ids.append(prob['_id'])

        d_qubo = {(i, j): val for [i, j, val] in prob['Q']}

        for emb_seed in range(10):
            print 'Generating embedding: ', emb_seed+1
            emb = utilities.get_embedding(prob['_id'], 'BAY1_2000Q', seed=emb_seed, adj=adj)

            for chain_strength in [10]:
                solver_params = {"answer_mode": "histogram", "num_reads": 10000, "num_spin_reversal_transforms": 100, 'chain_strength': chain_strength}
                emb_Q = utilities.apply_embedding(d_qubo, emb['chains'], adj, chain_strength=chain_strength)
                print 'Solving problem.'
                res = utilities.solve_dwave(solver, prob['_id'], emb_Q, emb['_id'], solver_name='BAY1_2000Q', **solver_params)
                result_ids.append(res['_id'])
                print 'Done solving!'

    exp_id = utilities.write_experiment(problem_ids, result_ids)
    print exp_id


def test_gnp_graphs():
    for i in range(5):
        G = utilities.nx.gnp_random_graph(20, .5, i)
        mis = utilities.nx.maximal_independent_set(G)
        print mis
        utilities.draw_mis(G, mis)


def test_gnp_hardness():
    exp_ids = list()
    rc = utilities.RemoteConnection(url, token)
    solver = rc.get_solver('BAY1_2000Q')
    # solver = utilities.local_connection.get_solver("c4-sw_optimize")
    adj = utilities.get_hardware_adjacency(solver)

    for p in np.arange(0.05, .45, 0.025):

        problem_ids = list()
        result_ids = list()

        print 'Starting experiment for p={}'.format(p)
        for i in range(50):
            params = {'n': 50, 'p': p}
            print 'Generating problem: ', i+1, ' for p= ', p
            prob = utilities.generate_problem('gnp', params, seed=i)
            problem_ids.append(prob['_id'])
            d_qubo = {(i, j): val for [i, j, val] in prob['Q']}

            print 'Generating embedding.'
            emb = utilities.get_embedding(prob['_id'], 'BAY1_2000Q', seed=0, adj=adj)

            solver_params = {"answer_mode": "histogram", "num_reads": 10000, "num_spin_reversal_transforms": 100, 'chain_strength': 10.}
            emb_Q = utilities.apply_embedding(d_qubo, emb['chains'], adj, chain_strength=10.)
            print 'Solving problem.'
            res = utilities.solve_dwave(solver, prob['_id'], emb_Q, emb['_id'], solver_name='BAY1_2000Q', **solver_params)
            result_ids.append(res['_id'])
            print 'Done solving!'

        exp_id = utilities.write_experiment(problem_ids, result_ids)
        exp_ids.append(exp_id)

    wrapper_id = utilities.wrap_experiments('test_gnp_hardness_.425', exp_ids)
    print 'Experiment wrapper ID: ', wrapper_id


def multiembed_gnp():
    exp_ids = list()
    rc = utilities.RemoteConnection(url, token)
    solver = rc.get_solver('BAY1_2000Q')
    adj = utilities.get_hardware_adjacency(solver)

    for i in range(50):

        problem_ids = list()
        result_ids = list()

        params = {'n': 50, 'p': .2}

        print 'Generating problem: ', i+1
        prob = utilities.generate_problem('gnp', params, seed=i)
        problem_ids.append(prob['_id'])

        d_qubo = {(i, j): val for [i, j, val] in prob['Q']}

        for emb_seed in range(25):
            print 'Generating embedding: ', emb_seed+1
            emb = utilities.get_embedding(prob['_id'], 'BAY1_2000Q', seed=emb_seed, adj=adj)

            for chain_strength in [10.]:
                solver_params = {"answer_mode": "histogram", "num_reads": 10000, "num_spin_reversal_transforms": 100, 'chain_strength': chain_strength}
                emb_Q = utilities.apply_embedding(d_qubo, emb['chains'], adj, chain_strength=chain_strength)
                print 'Solving problem.'
                res = utilities.solve_dwave(solver, prob['_id'], emb_Q, emb['_id'], solver_name='BAY1_2000Q', **solver_params)
                result_ids.append(res['_id'])
                print 'Done solving!'

        exp_id = utilities.write_experiment(problem_ids, result_ids)
        exp_ids.append(exp_id)
    utilities.wrap_experiments('multiembed_gnp_.2', exp_ids)


def test_dimacs():
    from pymongo import MongoClient
    client = MongoClient('localhost', 27017)
    db = client['Leiden_MIS']
    res_ids = list()
    chain_strength = 10.

    rc = utilities.RemoteConnection(url, token)
    solver = rc.get_solver('BAY1_2000Q')
    adj = utilities.get_hardware_adjacency(solver)

    params = {'filename': 'graphs/a265032_1dc.64.txt'}

    prob = utilities.generate_problem('benchmark', params, 0)
    p_id = prob['_id']

    print 'problem id: ', p_id
    d_qubo = {(i, j): val for [i, j, val] in prob['Q']}

    # emb = utilities.get_embedding(prob['_id'], 'BAY1_2000Q', seed=0, adj=adj)
    emb = utilities.best_embedding(prob['_id'], 'BAY1_2000Q', seed=0, adj=adj, num_attempts=20)

    emb_Q = utilities.apply_embedding(d_qubo, emb['chains'], adj, chain_strength=chain_strength)
    for trial in range(50):
        solver_params = {"answer_mode": "histogram", "num_reads": 10000, "num_spin_reversal_transforms": 100, 'chain_strength': chain_strength, 'trial': trial}
        print 'Solving problem (trial {})'.format(trial)
        res = utilities.solve_dwave(solver, prob['_id'], emb_Q, emb['_id'], solver_name='BAY1_2000Q', **solver_params)
        res_id = res['_id']
        res_ids.append(res_id)

    exp_id = utilities.write_experiment([p_id], res_ids)
    print 'Exp id: ', exp_id

    results = db['result'].find({'_id': {'$in': res_ids}})
    # print 'Min energy: ', min(res['energies'])
    # print 'Ind set: ', np.flatnonzero(np.array(res['solutions'][0]))
    mins = list()
    energies = list()
    for res in results:
        energies.extend(np.repeat(res['energies'], res['num_occurrences']))
        mins.append(min(res['energies']))

    gse = min(mins)
    print 'Minimum energy found: ', gse
    success_prob = sum(np.isclose(energies, gse)) / float(len(energies))
    print 'Success probability: ', success_prob


def random_graphs_networkx():
    exp_ids = list()

    params = {'p': .2}

    sc = SaaSClient('julia.dwavesys.local', '80')

    # for n in range(20, 65, 5):
    for n in range(60, 65, 5):
        print 'PROBLEM SIZE ', n
        params['n'] = n

        problem_ids = list()
        result_ids = list()

        for i in range(50):

            print 'Generating problem: ', i+1
            prob = utilities.generate_problem('gnp', params, seed=i)
            problem_ids.append(prob['_id'])

            # stuff for BMT solvers
            solver_params = dict(beta_start=.01, beta_end=2., num_sweeps=1000, num_reads=100)
            utilities.solve('SW', sc=sc, solver_name='dw_sa_cpu_gen', problem_id=prob['_id'], solver_params=solver_params)
            exit()
            # for trial in range(10):
            #     print 'Solving problem (trial {}).'.format(trial)
            #     res = utilities.solve('networkx', problem_id=prob['_id'], seed=trial, trial=trial)
            #     result_ids.append(res['_id'])
            #     print 'Done solving!'

        exp_id = utilities.write_experiment(problem_ids, result_ids)
        exp_ids.append(exp_id)
    utilities.wrap_experiments('random_graphs_p=.2_60_networkx', exp_ids)


def random_graphs_full():
    rc = utilities.RemoteConnection(url, token)
    solver = rc.get_solver('BAY1_2000Q')
    adj = utilities.get_hardware_adjacency(solver)

    exp_ids = list()

    params = {'p': .2}

    for n in range(20, 65, 5):
        print 'PROBLEM SIZE ', n
        params['n'] = n

        for i in range(50):
            problem_ids = list()
            result_ids = list()

            print 'Generating problem: ', i+1
            prob = utilities.generate_problem('gnp', params, seed=i)
            problem_ids.append(prob['_id'])

            d_qubo = {(i, j): val for [i, j, val] in prob['Q']}

            emb = utilities.best_embedding(prob['_id'], 'BAY1_2000Q', 0, adj, 10)

            for chain_strength in [2., 3., 4., 5., 6., 7., 8., 9., 10.]:
                emb_Q = utilities.apply_embedding(d_qubo, emb['chains'], adj, chain_strength=chain_strength)
                # for trial in range(10):
                for trial in range(1):
                    solver_params = {"answer_mode": "histogram", "num_reads": 10000, "num_spin_reversal_transforms": 100, 'chain_strength': chain_strength, 'trial': trial}
                    print 'Solving problem (trial {}, chain strength {}).'.format(trial, chain_strength)
                    res = utilities.solve_dwave(solver, prob['_id'], emb_Q, emb['_id'], solver_name='BAY1_2000Q', **solver_params)
                    result_ids.append(res['_id'])
                    # print res['_id']
                    print 'Done solving!'

            exp_id = utilities.write_experiment(problem_ids, result_ids)
            exp_ids.append(exp_id)
    utilities.wrap_experiments('random_graphs_p=.2_60', exp_ids)


def test_saas():
    from SaaSClient import SaaSClient

    sc = SaaSClient('julia.dwavesys.local', '80')
    # sc = SaaSClient('syarkoni-t430s', '8081')
    solution = sc.submit([0, 0], {(0, 1): -1}, 'dw_sa_cpu_gen', beta_start=.01, beta_end=2., num_sweeps=1000, num_reads=100, db_host='julia.dwavesys.local:27017')
    # solution = sc.submit([], {}, [])
    # solver_list = sc.get_solvers()
    # print solver_list
    print solution

def random_graphs_sa():
    exp_ids = list()

    params = {'p': .2}

    sc = SaaSClient('julia.dwavesys.local', '80')
    # sc = SaaSClient('markov', '80')

    for n in range(20, 65, 5):
        print 'PROBLEM SIZE ', n
        params['n'] = n

        problem_ids = list()
        result_ids = list()

        for i in range(50):
            print 'Generating problem: ', i+1
            prob = utilities.generate_problem('gnp', params, seed=i)
            problem_ids.append(prob['_id'])
            for num_sweeps in [10, 100, 1000]:
                solver_params = dict(beta_start=.01, beta_end=3., num_sweeps=num_sweeps, num_reads=1000)
                res = utilities.solve('SW', sc=sc, solver_name='dw_sa_cpu_gen', problem_id=prob['_id'], solver_params=solver_params)
                # res = utilities.solve('SW', sc=sc, solver_name='dw_lta_cpu_gen', problem_id=prob['_id'], solver_params=solver_params)
                print res['_id']
                result_ids.append(res['_id'])

        exp_id = utilities.write_experiment(problem_ids, result_ids)
        exp_ids.append(exp_id)
    utilities.wrap_experiments('random_graphs_p=.2_60_SA', exp_ids)

def test_tsp():
    import matplotlib.pyplot as plt

    h = [1.02134375, 1.01101875, 1.010325, 1.01135625, 1.0095875, 1.01205625, 1.01425, 1.01031875, 1.02456875]
    J = {(4, 8): 0.00444375, (3, 0): 0.5, (0, 7): 0.005875, (3, 7): 0.00444375, (8, 5): 0.5, (7, 6): 0.5, (6, 3): 0.5, (1, 5): 0.00514375, (0, 4): 0.00514375, (8, 6): 0.5, (4, 1): 0.5, (5, 4): 0.5, (8, 2): 0.5, (7, 1): 0.5, (6, 0): 0.5, (2, 1): 0.5, (8, 7): 0.5, (1, 0): 0.5, (5, 3): 0.5, (7, 4): 0.5, (2, 0): 0.5, (1, 8): 0.005875, (4, 3): 0.5, (5, 2): 0.5}
    print len(h)
    vars = set()
    for (i, j) in J.keys():
        vars.add(i)
        vars.add(j)
    print vars, len(vars)
    print J.keys()
    import networkx as nx
    g = nx.Graph()
    for (i, j) in J.keys():
        g.add_edge(i+1, j+1)
    nx.draw(g)
    # plt.show()
    # exit()
    # exit()
    sc = SaaSClient('julia.dwavesys.local', '80')
    # sc = SaaSClient('syarkoni-t430s', '8081')
    solution = sc.submit(h, J, 'dw_sa_cpu_gen', beta_start=.01, beta_end=3., num_sweeps=100, num_reads=10, db_host='julia.dwavesys.local:27017')
    # solution = sc.submit([2.391539750137398597592502, 0, 0], {(0, 1): -1, (1, 2): -1}, 'dw_sa_cpu_gen', beta_start=.1, beta_end=3., num_sweeps=1000, num_reads=100, db_host='julia.dwavesys.local:27017')
    # solution = sc.submit([], {}, [])
    # solver_list = sc.get_solvers()
    # print solver_list
    print solution

def print_mask():
    from bmf.mask.util import to_eps
    from bmf.mask import Mask
    m = Mask.from_solver('BAY1_2000Q')
    print m.num_qubits
    print len(m.qubits)
    # eps = to_eps(m, vertex_radius=4)
    # with open("dw2000q.eps", 'w') as f:
    #     f.write(eps)


def test_anneal_offsets_framework():
    rc = utilities.RemoteConnection(url, token)
    solver_name = 'DW_2000Q_2'
    solver = rc.get_solver(solver_name)
    adj = utilities.get_hardware_adjacency(solver)

    problem_sizes = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
    size_to_chain_strength = {20: 2, 25: 2, 30: 2, 35: 2, 40: 3, 45: 3, 50: 4, 55: 5, 60: 7, 65: 10, 70: 12}
    exp_ids = list()

    params = {'p': .2}
    for num_problems in [50]:
        for n in problem_sizes:
            print 'PROBLEM SIZE ', n
            params['n'] = n

            for i in range(0, num_problems):
                problem_ids = list()
                result_ids = list()

                print 'Generating problem: ', i+1
                prob = utilities.generate_problem('gnp', params, seed=i)
                problem_ids.append(prob['_id'])

                d_qubo = {(i, j): val for [i, j, val] in prob['Q']}

                emb = utilities.best_embedding(prob['_id'], solver_name, 0, adj, 10)
                chain_strength = size_to_chain_strength[n]

                # for chain_strength in [4.]:
                emb_Q = utilities.apply_embedding(d_qubo, emb['chains'], adj, chain_strength=chain_strength)
                # for trial in range(10):
                for trial in range(1):
                    print 'Solving problem (trial {}, chain strength {}).'.format(trial, chain_strength)
                    # inital parameters = 0, no restarts
                    solver_params = {"answer_mode": "histogram", "num_reads": 10000, "num_spin_reversal_transforms": 1, 'chain_strength': chain_strength, 'trial': trial, 'tuner': 'cma', 'tune': True, 'annealing_time': 1, 'answer_mode': 'raw'}
                    res = utilities.solve_dwave(solver, prob['_id'], emb_Q, emb['_id'], solver_name=solver_name, **solver_params)
                    result_ids.append(res['_id'])
                    print 'finished inital parameters = 0, no restarts'
                    solver_params = {"answer_mode": "histogram", "num_reads": 50000, "num_spin_reversal_transforms": 100, 'chain_strength': chain_strength, 'trial': trial, 'tuner': 'cma', 'tune': False, 'annealing_time': 1, 'answer_mode': 'raw'}
                    res = utilities.solve_dwave(solver, prob['_id'], emb_Q, emb['_id'], solver_name=solver_name, **solver_params)
                    result_ids.append(res['_id'])
                    print 'finished QPU run with said params'
                    # initial parameters = uniform, no restarts
                    solver_params = {"answer_mode": "histogram", "num_reads": 10000, "num_spin_reversal_transforms": 1, 'chain_strength': chain_strength, 'trial': trial, 'tuner': 'cma', 'tune': True, 'tuner_params': {'initial_offsets': 'uniform'}, 'annealing_time': 1, 'answer_mode': 'raw'}
                    res = utilities.solve_dwave(solver, prob['_id'], emb_Q, emb['_id'], solver_name=solver_name, **solver_params)
                    result_ids.append(res['_id'])
                    print 'finished initial parameters = uniform, no restarts'
                    solver_params = {"answer_mode": "histogram", "num_reads": 50000, "num_spin_reversal_transforms": 100, 'chain_strength': chain_strength, 'trial': trial, 'tuner': 'cma', 'tune': False, 'tuner_params': {'initial_offsets': 'uniform'}, 'annealing_time': 1, 'answer_mode': 'raw'}
                    res = utilities.solve_dwave(solver, prob['_id'], emb_Q, emb['_id'], solver_name=solver_name, **solver_params)
                    result_ids.append(res['_id'])
                    print 'finished QPU run with said params'
                    # no tuning
                    solver_params = {"answer_mode": "histogram", "num_reads": 50000, "num_spin_reversal_transforms": 100, 'chain_strength': chain_strength, 'trial': trial, 'tuner': 'None', 'annealing_time': 1, 'answer_mode': 'raw'}
                    res = utilities.solve_dwave(solver, prob['_id'], emb_Q, emb['_id'], solver_name=solver_name, **solver_params)
                    result_ids.append(res['_id'])
                    # print res['_id']
                    print 'Done solving!'

                exp_id = utilities.write_experiment(problem_ids, result_ids)
                exp_ids.append(exp_id)
        utilities.wrap_experiments('random_graphs_p=.2_CMA_num_problems={}_10k_tune_50k_run'.format(problem_sizes), exp_ids)


def test_anneal_offsets():
    from qpu_param_es import dr1_es
    from cholesky_cma_es import cholesky_cma_es
    import itertools
    print 'trying to connect to SAPI'
    rc = utilities.RemoteConnection(url, token)
    print 'connected.'
    solver = rc.get_solver('DW_2000Q_1')
    offset_ranges_all_qbs = np.array(solver.properties['anneal_offset_ranges'])
    param_offset_ranges = np.zeros((len(offset_ranges_all_qbs),))

    adj = utilities.get_hardware_adjacency(solver)

    params = {'p': .2}
    trial = 0
    for n in range(50, 51, 5):
        print 'PROBLEM SIZE ', n
        params['n'] = n

        for i in range(2, 3):
            problem_ids = list()

            print 'Generating problem: ', i+1
            prob = utilities.generate_problem('gnp', params, seed=i)
            problem_ids.append(prob['_id'])

            d_qubo = {(i, j): val for [i, j, val] in prob['Q']}

            emb = utilities.best_embedding(prob['_id'], 'BAY14_2000Q', 0, adj, 1)
            problem_qubits = list(itertools.chain.from_iterable(emb['chains']))
            print 'max chain length = ', max(map(len, emb['chains'])), np.mean(map(len, emb['chains']))
            print 'total size of embedding', sum(map(len, emb['chains']))
            # exit()

            # OK we got the upper- lower-bounds for every qubit. Now we have to make them on a per-chain basis as per
            # the embedding.

            # ub = np.array(map(lambda x: x[1], offset_ranges_all_qbs[problem_qubits]))
            # lb = np.array(map(lambda x: x[0], offset_ranges_all_qbs[problem_qubits]))
            ub = np.array(map(lambda x: x[1], offset_ranges_all_qbs))
            lb = np.array(map(lambda x: x[0], offset_ranges_all_qbs))


            chain_ub = list()
            chain_lb = list()

            for chain in emb['chains']:
                chain_ub.append( min(ub[chain]) )
                chain_lb.append( max(lb[chain]) )

            num_samples=1000

            for chain_strength in [5.]:
                emb_Q = utilities.apply_embedding(d_qubo, emb['chains'], adj, chain_strength=chain_strength)
                # for trial in range(10):
                def ff(x):
                    for i, offset in enumerate(x):
                        # param_offset_ranges[problem_qubits] = x
                        param_offset_ranges[emb['chains'][i]] = offset

                    # for i in range(len(x)):
                    #     try:
                    #         assert chain_lb[i] < x[i] < chain_ub[i]
                    #     except AssertionError as e:
                    #         print i, chain_lb[i], x[i], chain_ub[i]
                    #         print emb['chains'][i]
                    #         exit()

                    # print 'Solving problem on QPU...'
                    # for trial in range(10):
                    solver_params = {"answer_mode": "histogram", "num_reads": num_samples, "num_spin_reversal_transforms": 1, 'chain_strength': chain_strength, 'trial': trial, 'anneal_offsets': param_offset_ranges.tolist()}

                    # print 'Solving problem (trial {}, chain strength {}).'.format(trial, chain_strength)

                    res = utilities.solve_dwave(solver, prob['_id'], emb_Q, emb['_id'], solver_name='BAY14_2000Q', **solver_params)
                    # print np.percentile(res['energies'], 25), min(res['energies'])
                    # return np.percentile(res['energies'], 25)
                    return min(res['energies'])
                # dr1_results = dr1_es(n, 10000, ff, chain_lb, chain_ub, num_samples)
                cma_results = cholesky_cma_es(n, 100000, ff, chain_lb, chain_ub, num_samples)

                import json
                with open('cma_es_test_chain_n={}_50_p{}_min_1000samples.txt'.format(n, i), 'w') as f:
                    json.dump(cma_results[-1], f)


    # rc = utilities.RemoteConnection(url, token)
    # solver = rc.get_solver('BAY14_2000Q')
    # print np.shape(solver.properties['anneal_offset_ranges'])

if __name__ == '__main__':
    # print_mask()
    test_anneal_offsets_framework()
    # test_anneal_offsets()
    exit()
    # multiembed_gnp()
    # test_dimacs()
    # random_graphs_networkx()
    random_graphs_sa()
