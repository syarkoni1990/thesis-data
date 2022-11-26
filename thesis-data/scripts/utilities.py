__author__ = 'syarkoni'

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import itertools


from dwave_sapi2.local import local_connection
from dwave_sapi2.core import solve_ising, solve_qubo, async_solve_ising, await_completion
from dwave_sapi2.util import ising_to_qubo, qubo_to_ising, get_hardware_adjacency
from dwave_sapi2.embedding import find_embedding, embed_problem, unembed_answer
from dwave_sapi2.remote import RemoteConnection

from cholesky_cma_es import cholesky_cma_es

from pymongo import MongoClient, ASCENDING, DESCENDING
from bson import ObjectId
from collections import defaultdict

class TemporarySeed:
    """Class for holding the random seed while a given seed is used temporarily."""
    def __init__(self, seed):
        self.state = None
        self.numpy_state = None
        self.seed = seed

    def __enter__(self):
        self.state = random.getstate()
        self.numpy_state = np.random.get_state()
        random.seed(self.seed)
        np.random.seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        random.setstate(self.state)
        np.random.set_state(self.numpy_state)


type_to_class = {"<type 'list'>": [], "<type 'dict'>": defaultdict(int)}

client = MongoClient('localhost', 27017)
# db = client['Leiden_MIS_CMA_test']
db = client['Leiden_MIS_tuning_paper']

problem_attrs = ['model', 'params', 'seed', 'Q', 'num_vars']
embedding_attrs = ['problem_id', 'solver', 'chains', 'seed']


def assert_valid(query, col):
    if col == 'problem':
        for attr in problem_attrs:
            assert attr in query
        return True

    if col == 'embedding':
        for attr in embedding_attrs:
            assert attr in query
        return True

    raise ValueError('No collection matching {}.'.format(col))


def find(query, col):
    res = db[col].find_one(query)
    if res is None:
        res = db[col].insert(query)
        return res

    return res['_id']


def generate_mis_qubo(G):
    num_vars = len(G.nodes())
    Q = [[0 for _ in range(num_vars)] for _ in range(num_vars)]

    # Sheir's way:
    for node in G.nodes():
        Q[node][node] = -1
    for (i, j) in G.edges():
        Q[i][j] = 2

    return np.vstack(Q)


def apply_embedding(d_qubo, chains, adj, chain_strength):
    h, J, _ = qubo_to_ising(d_qubo)

    hemb, jemb, jchain, embedding = embed_problem(h, J, chains, adj)

    high = max(map(abs, jemb.values()))
    for (i, j) in jchain.keys():
        jemb[(i, j)] = -high * chain_strength

    emb_d_qubo, _ = ising_to_qubo(hemb, jemb)

    return emb_d_qubo


def generate_embedding(problem_id, solver_name, adj, seed=0):
    """
    Takes a QUBO problem, solver, seed and returns an embedding.
    :param problem:
    :param solver_name:
    :param adj:
    :param seed:
    :return:
    """

    problem = db['problem'].find_one({'_id': problem_id})

    with TemporarySeed(seed):

        d_emb = {'solver': solver_name, 'problem_id': problem_id, 'seed': seed}

        d_qubo = {(i, j): val for [i, j, val] in problem['Q']}

        emb = find_embedding(d_qubo, adj, tries=25)
        if not emb:
            raise ValueError('Could not embed.')

        if problem['num_vars']!=len(emb):
            raise ValueError("Something went wrong with the embedding. num_vars={} but len(emb)={}".format(problem['num_vars'], len(emb)))


        d_emb['chains'] = emb

        emb_id = find(d_emb, 'embedding')

        d_emb['_id'] = emb_id

    return d_emb


def get_embedding(problem_id, solver_name, seed=0, adj=None):
    d_emb = {'solver': solver_name, 'problem_id': problem_id, 'seed': seed}
    emb = db['embedding'].find_one(d_emb)
    if emb is not None:
        return emb

    assert adj is not None

    return generate_embedding(problem_id, solver_name, adj, seed)


def calc_chain_statistics(solutions, chains):
    solutions = np.array(solutions)
    chain_sols = [solutions[:, chain] for chain in chains]
    chain_sols = [np.mean(sols, axis=1) for sols in chain_sols]
    chain_sols = [[int(s == 0 or s == 1) for s in sol] for sol in chain_sols]

    chain_stats = {'chain_intact': chain_sols}

    return chain_stats


def get_solver_search_name(solver_name, params):
    if '_sa_' in solver_name or '_lta_' in solver_name:
        return '{}:_beta_start{}_beta_end{}_num_sweeps{}_num_reads{}'.format(solver_name, params['beta_start'], params['beta_end'], params['num_sweeps'], params['num_reads'])

    raise NotImplementedError("Only Simulated Annealing parameters are supported for BMT solvers.")


def custom_qubo_to_ising(q):
    """
    (h, J, offset) = qubo_to_ising(q)

    Map a binary quadratic program x' * Q * x defined over 0/1 variables to
    an ising model defined over -1/+1 variables. We return the h and J
    defining the Ising model as well as the offset in energy between the
    two problem formulations, i.e. x' * Q * x = offset + s' * J * s + h' * s. The
    linear term of the qubo is contained along the diagonal of Q.

    See ising_to_qubo(h, J) for the inverse function.

    Args:
        q: Q for qubo problem

    Returns:
        A tuple which contains a list h for ising problem, a dictionary J
        for ising problem and offset

    Raises:
        ValueError: bad parameter value
    """
    hdict = defaultdict(float)
    j = {}
    offset = 0
    all_nodes = set()
    for (i, k), e in q.iteritems():
        if i == k:
            all_nodes.add(i)
            hdict[i] += 0.5 * e
            offset += 0.5 * e
        else:
            j[(i, k)] = 0.25 * e
            hdict[i] += 0.25 * e
            hdict[k] += 0.25 * e
            offset += 0.25 * e

    hdict = dict((k, v) for k, v in hdict.iteritems())
    if hdict:
        h = [0] * (1 + max(hdict))
    else:
        h = []
    for i, v in hdict.iteritems():
        h[i] = v
    j = dict((k, v) for k, v in j.iteritems() if v!=0)

    return h, j, offset


def solve_sw_solver(sc, solver_name, problem_id, **params):
    assert solver_name is not None

    trial = params.pop('trial', 0)
    solver_search_name = get_solver_search_name(solver_name, params['solver_params'])
    res = {'problem_id': problem_id, 'solver_name': solver_search_name, 'trial': trial}

    found_res = db['result'].find_one(res)

    if found_res is not None:
        return found_res

    prob = db['problem'].find_one({'_id': problem_id})
    solver_params = params.pop('solver_params')
    assert solver_name is not None
    d_qubo = {(i, j): k for [i, j, k] in prob['Q']}

    Q = [[0 for _ in range(prob['num_vars'])] for _ in range(prob['num_vars'])]
    for (i, j), k in d_qubo.iteritems():
        Q[i][j] = k

    nodes = set()
    for (i, j), k in d_qubo.iteritems():
        if i==j:
            nodes.add(i)

    # h, J, _ = qubo_to_ising(d_qubo)
    h, J, _ = custom_qubo_to_ising(d_qubo)

    submitted = False

    while not submitted:
        try:
            answers = sc.submit(h, J, solver_name, db_host='julia.dwavesys.local:27017', **solver_params)[solver_name]
            submitted = True
        except:
            print 'Could not connect to julia... trying again in 5 seconds.'
            time.sleep(5)

    sols = np.array(answers['solutions']) > 0
    sols = np.array(sols, dtype=int).tolist()

    res['solutions'] = sols
    res['energies'] = [eval_qubo(Q, x) for x in sols]
    res['num_occurrences'] = answers['num_occurrences']
    res['timing'] = answers['timing']
    res['order'] = answers.get('order', range(len(sols)))
    best_energy = min(res['energies'])
    if best_energy < prob.get('record_energy', np.inf):
        db['problem'].update({'_id': problem_id}, {'$set': {'record_energy': best_energy}})

    res_id = find(res, 'result')
    res['_id'] = res_id

    return res


def solve(solver_type, **params):
    if solver_type == 'DWAVE':
        return solve_dwave(**params)

    if solver_type == 'networkx':
        return solve_networkx(**params)

    if solver_type == 'SW':
        return solve_sw_solver(**params)

    raise ValueError('Solver {} not available.'.format(solver_type))


def solve_networkx(problem_id, seed, **params):
    import networkx as nx

    trial = params.pop('trial', 0)

    res = {'problem_id': problem_id, 'solver_name': 'networkx', 'embedding_id': None, 'trial': trial}

    found_res = db['result'].find_one(res)

    if found_res is not None:
        return found_res

    with TemporarySeed(seed):
        prob = db['problem'].find_one({'_id': problem_id})
        G = nx.Graph()

        Q = [[0 for _ in range(prob['num_vars'])] for _ in range(prob['num_vars'])]

        for i, j, val in prob['Q']:
            if i != j:
                G.add_edge(i+1, j+1)
            else:
                G.add_node(i+1)
            Q[i][j] = val

        t = time.time()
        mis = nx.maximal_independent_set(G)
        t = time.time()-t

        mis = np.array(mis) - 1

        sol = np.array([0 for _ in G.nodes()])
        sol[mis] = 1

        energy = eval_qubo(Q, sol)

        res['solutions'] = [sol.tolist()]
        res['energies'] = [energy]
        res['num_occurrences'] = [1]
        res['total_cpu_time'] = t
        if energy< prob.get('record_energy', np.inf):
            db['problem'].update({'_id': problem_id}, {'$set': {'record_energy': energy}})

        res_id = find(res, 'result')
        res['_id'] = res_id

    return res


def apply_offsets(num_vars, vars_list, anneal_offsets):
    param_offset_ranges = np.zeros((num_vars,))

    for i, offset in enumerate(anneal_offsets):
        param_offset_ranges[vars_list[i]] = offset

    return param_offset_ranges.tolist()


def _cma_f(solver, d_qubo, embedding, **params):
    cma_params = params.pop('tuner_params', {})

    offset_ranges_all_qbs = np.array(solver.properties['anneal_offset_ranges'])
    num_qubits = int(solver.properties['num_qubits'])

    _num_vars = len(embedding['chains'])
    vars_list = embedding['chains']

    # param_offset_ranges = np.zeros((len(offset_ranges_all_qbs),))

    # Get the upper- lower-bounds for every qubit
    ub = np.array(map(lambda x: x[1], offset_ranges_all_qbs))
    lb = np.array(map(lambda x: x[0], offset_ranges_all_qbs))

    chain_ub = list()
    chain_lb = list()

    # Bound the offsets for each chain by the bounds of the qubits
    for chain in embedding['chains']:
        chain_ub.append( min(ub[chain]) )
        chain_lb.append( max(lb[chain]) )

    num_samples = params['num_reads']
    assert num_samples > 100
    params['num_reads'] = 100
    params['num_spin_reversal_transforms'] = 1
    params['answer_mode'] = 'raw'

    res_list = defaultdict(list)

    def ff(x):
        params['anneal_offsets'] = apply_offsets(num_qubits, vars_list, x)
        # params['anneal_offsets'] = param_offset_ranges.tolist()
        # print 'submitting problem...'
        res = solve_qubo(solver, d_qubo, **params)
        # print 'done.'

        for k in res.keys():
            res_list[k].append(res[k])
        # print 'how many res collected: ', len(res_list[res.keys()[0]])
        # print res['solutions']
        # print min(res['energies']), np.mean(res['energies'])
        # print 'how many mins: ', sum(np.flatnonzero(np.array(res['energies']) == min(res['energies'])))
        # return min(res['energies'])
        print 'finished one'
        return np.mean(res['energies'])

    cma_results = cholesky_cma_es(_num_vars, num_samples, ff, chain_lb, chain_ub, 100, **cma_params)

    return res_list, cma_results

def _solve_dwave_with_tuner(solver, problem_id, emb_id, d_qubo, tuner, tuner_params, **params):
    assert tuner == 'cma'

    # tuner_params = dict()
    # tuner_params = params.get('tuner_params', {})
    # tuner_params['with_restarts'] = params.pop('with_restarts', False)
    # tuner_params['initial_offsets'] = params.pop('initial_offsets', None)

    params['tuner_params'] = tuner_params

    tuner_dict = {
        'tuner': tuner,
        'problem_id': problem_id,
        'embedding_id': emb_id,
        'tune': True,
        'with_restarts': tuner_params['with_restarts'],
        'initial_offsets': tuner_params['initial_offsets'],
        'tuner_run': tuner_params['tuner_run'],
    }

    if emb_id is not None:
        emb = db['embedding'].find_one({'_id': emb_id})
    else:
        emb = dict()
        emb['chains'] = solver.properties['qubits']

    res, tuner_data = _cma_f(solver, d_qubo, emb, **params)

    for k in tuner_data.keys():
        try:
            tuner_dict[k] = tuner_data[k].tolist()
        except:
            tuner_dict[k] = tuner_data[k]

    merged_res = {'timing': defaultdict(int), 'solutions': list(), 'energies': list()}

    for k in ['timing', 'solutions', 'energies']:
        for e in res[k]:
            if type(e) == dict:
                for kk in e.keys():
                    if 'per_sample' in kk or 'per_run' in kk:
                        merged_res[k][kk] = e[kk]
                    else:
                        merged_res[k][kk] += e[kk]
            else:
                merged_res[k] += e

    merged_res['timing'] = dict(merged_res['timing'])
    return merged_res, tuner_dict


def solve_dwave(solver, problem_id, d_qubo, emb_id=None, **params):
    # Grab the D-Wave solver name
    solver_name = params.pop('solver_name')
    assert solver_name is not None

    trial = params.pop('trial', 0)
    res = {'problem_id': problem_id, 'solver_name': solver_name, 'embedding_id': emb_id, 'trial': trial}
    if 'chain_strength' in params:
        chain_strength = params.pop('chain_strength')
        res['chain_strength'] = chain_strength
    else:
        chain_strength = None

    # Name of the param tuning algorithm (right now only CMA is supported)
    tuner = params.pop('tuner')
    assert tuner is 'None' or tuner == 'cma'

    # If 'tune' is True, this is a learning procedure. Otherwise, grab the tuned parameters and move on.
    tune = params.pop('tune', False if tuner is None else True)

    # grab tuner_id or generate it (either ObjectId or None)
    if tuner != 'None':
        tuner_params = params.pop('tuner_params', {})
        tuner_params['with_restarts'] = tuner_params.get('with_restarts', False)
        tuner_params['initial_offsets'] = tuner_params.get('initial_offsets', None)
        tuner_params['tuner_run'] = tuner_params.get('tuner_run', 0)

        tuner_dict = {
            'tuner': tuner,
            'problem_id': problem_id,
            'embedding_id': emb_id,
            'with_restarts': tuner_params['with_restarts'],
            'initial_offsets': tuner_params['initial_offsets'],
            'tuner_run': tuner_params['tuner_run'],
            'chain_strength': chain_strength,
        }

        tuner_doc = db['tuner'].find_one(tuner_dict)
        if not tune:
            res['num_reads'] = params['num_reads']
            if tuner_doc is None:
                raise RuntimeError('Must tune tuner before first use!!')

            res['tuner_id'] = tuner_doc['_id']
            res['tune'] = False

            found_res = db['result'].find_one(res)

            if found_res is not None:
                return found_res

            anneal_offsets = tuner_doc['opt_param'][-1]

            if emb_id:
                emb = db['embedding'].find_one({'_id': emb_id})
                vars_list = emb['chains']
            else:
                vars_list = range(solver.properties['num_qubits'])

            params['anneal_offsets'] = apply_offsets(solver.properties['num_qubits'], vars_list, anneal_offsets)
            print params['num_reads']
            answers = solve_qubo(solver, d_qubo, **params)
            # answers, tuner_doc = _solve_dwave_with_tuner(solver, problem_id, emb_id, d_qubo, tuner, **params)
        else:
            res['tune'] = True
            res['num_reads'] = params['num_reads']-100
            if tuner_doc is not None:
                res['tuner_id'] = tuner_doc['_id']
                found_res = db['result'].find_one(res)
                assert found_res is not None
                return found_res


            answers, tuner_doc = _solve_dwave_with_tuner(solver, problem_id, emb_id, d_qubo, tuner, tuner_params, **params)
            tuner_doc['chain_strength'] = chain_strength

            res['tuner_id'] = find(tuner_doc, 'tuner')

    else:
        res['tuner_id'] = 'None'
        res['num_reads'] = params['num_reads']
        found_res = db['result'].find_one(res)

        if found_res is not None:
            return found_res

        answers = solve_qubo(solver, d_qubo, **params)

    # res['tuner_id'] = tuner_doc['_id']
    # res = {'problem_id': problem_id, 'solver_name': solver_name, 'embedding_id': emb_id, 'trial': trial, 'tuner': tuner, 'tune': tune}


    # found_res = db['result'].find_one(res)
    # if found_res is not None:
    #     return found_res

    # answers = solve_qubo(solver, d_qubo, **params)

    # if tuner:
    #     # tuner_dict = {
    #     #     'tuner': tuner,
    #     #     'problem_id': problem_id,
    #     #     'embedding_id': emb_id,
    #     #     'with_restarts': tuner_params['with_restarts'],
    #     #     'initial_offsets': tuner_params['initial_offsets'],
    #     # }
    #     #
    #     # if not tune:
    #     #     tuner_doc = db['tuner'].find_one(tuner_dict)
    #     #     if tuner_doc is None:
    #     #         raise RuntimeError('Must tune tuner before first use!!')
    #     #
    #     #     anneal_offsets = tuner_doc['opt_param'][-1]
    #     #
    #     #     if emb_id:
    #     #         emb = db['embedding'].find_one({'_id': emb_id})
    #     #         vars_list = emb['chains']
    #     #     else:
    #     #         vars_list = range(solver.properties['num_qubits'])
    #     #     params['anneal_offsets'] = apply_offsets(solver.properties['num_qubits'], vars_list, anneal_offsets)
    #         answers = solve_qubo(solver, d_qubo, **params)
    #     else:
    #         # answers, tuner_doc = _solve_dwave_with_tuner(solver, problem_id, emb_id, d_qubo, tuner, **params)
    #
    #     res['tuner_id'] = find(tuner_doc, 'tuner')
    #
    # else:
    #     answers = solve_qubo(solver, d_qubo, **params)

    prob = db['problem'].find_one({'_id': problem_id})

    if emb_id is not None:
        assert chain_strength is not None

        emb = db['embedding'].find_one({'_id': emb_id})
        unembeded_sols = unembed_answer(answers['solutions'], emb['chains'], broken_chains='vote')


        unembeded_sols = ((np.array(unembeded_sols)+1)/2).tolist()

        num_vars = len(emb['chains'])

        original_Q = prob['Q']
        Q = [[0 for _ in range(num_vars)] for _ in range(num_vars)]
        for [i, j, val] in original_Q:
            Q[i][j] = val

        res['chain_statistics'] = calc_chain_statistics(answers['solutions'], emb['chains'])

        answers['solutions'] = unembeded_sols
        answers['energies'] = [eval_qubo(Q, x) for x in unembeded_sols]

    res['solutions'] = answers['solutions']
    res['energies'] = answers['energies']
    res['timing'] = answers['timing']
    try:
        res['num_occurrences'] = answers['num_occurrences']
    except:
        res['num_occurrences'] = [1] * len(res['solutions'])
    best_energy = min(answers['energies'])
    if best_energy < prob.get('record_energy', np.inf):
        db['problem'].update({'_id': problem_id}, {'$set': {'record_energy': best_energy}})

    res_id = find(res, 'result')
    res['_id'] = res_id

    return res


def generate_problem(model, params, seed):
    if model == 'BA':
        assert 'm' in params
        assert 'n' in params

        d_prob = {'model': 'BA', 'params': params, 'seed': seed}
        prob = db['problem'].find_one(d_prob)
        if prob is not None:
            return prob

        G = nx.barabasi_albert_graph(params['n'], params['m'], seed)
        Q = generate_mis_qubo(G)
        d_prob['num_vars'] = len(Q)
        condensed_Q = []
        for i in range(len(Q)):
            for j in range(len(Q)):
                if Q[i][j] != 0.:
                    condensed_Q.append([i, j, Q[i][j]])
        d_prob['Q'] = condensed_Q
        prob_id = find(d_prob, 'problem')
        d_prob['_id'] = prob_id

    elif model == 'gnp':
        assert 'n' in params
        assert 'p' in params

        d_prob = {'model': 'gnp', 'params': params, 'seed': seed}
        prob = db['problem'].find_one(d_prob)
        if prob is not None:
            return prob


        G = nx.gnp_random_graph(params['n'], params['p'], seed=seed, directed=False)
        Q = generate_mis_qubo(G)
        d_prob['num_vars'] = len(Q)
        condensed_Q = []
        for i in range(len(Q)):
            for j in range(len(Q)):
                if Q[i][j] != 0.:
                    condensed_Q.append([i, j, Q[i][j]])
        d_prob['Q'] = condensed_Q
        prob_id = find(d_prob, 'problem')
        d_prob['_id'] = prob_id

    elif model == 'benchmark':
        assert 'filename' in params

        filename = params['filename']

        d_prob = {'model': 'benchmark', 'params': params}
        prob = db['problem'].find_one(d_prob)
        if prob is not None:
            return prob

        G = convert_dimacs_file_to_nx(filename)
        Q = generate_mis_qubo(G)
        d_prob['num_vars'] = len(Q)
        condensed_Q = []
        for i in range(len(Q)):
            for j in range(len(Q)):
                if Q[i][j] != 0.:
                    condensed_Q.append([i, j, Q[i][j]])
        d_prob['Q'] = condensed_Q
        prob_id = find(d_prob, 'problem')
        d_prob['_id'] = prob_id

    else:
        raise ValueError("Model {} not recognized!".format(model))

    return d_prob


def find_max_ind_set_dwave(G, local=True):

    if local:
        solver = local_connection.get_solver("c4-sw_optimize")
        params = {"answer_mode": "histogram", "num_reads": 1000}
    else:
        url = 'https://dw2x.alpha.dwavesys.com/sapi/'
        token = 'BMT-383efe5197b74aed84c07de22e84fe6cae6c28c4'
        rc = RemoteConnection(url, token)
        # solver = rc.get_solver('BAY1_2000Q_VFYC')
        solver = rc.get_solver('BAY1_2000Q')
        params = {"answer_mode": "histogram", "num_reads": 10000, "num_spin_reversal_transforms": 100}

    adj = get_hardware_adjacency(solver)

    Q = generate_mis_qubo(G)

    d_qubo = {(i, j): Q[i][j] for i in range(len(Q)) for j in range(len(Q))}

    h, J, _ = qubo_to_ising(d_qubo)

    emb = find_embedding(J, adj)

    if len(emb) == 0:
        raise RuntimeError("No embedding found.")
    else:
        pass
        print 'SIZE OF EMB: ', sum(map(len, emb))
        print 'LONGEST CHAIN: ', max(map(len, emb))

    hemb, jemb, jchain, embedding = embed_problem(h, J, emb, adj)

    high = max(map(abs, jemb.values()))
    chain_strength = 12
    for (i, j) in jchain.keys():
        jemb[(i, j)] = -high * chain_strength

    answer = solve_ising(solver, hemb, jemb, **params)

    gse = min(answer['energies'])
    states = np.array(unembed_answer(answer['solutions'], emb, h=hemb, j=jemb))

    states = states[np.flatnonzero(np.isclose(answer['energies'], gse))]
    states = map(lambda x: np.array(x>0, dtype=int).tolist(), states)

    mis = np.flatnonzero(np.array(states[0]))

    return mis.tolist()


def find_max_ind_set_networkx(G):
    t = time.time()
    mis = nx.maximal_independent_set(G)
    t -= time.time()
    t *= -1
    # print t
    return mis


def eval_qubo(Q, x):
    return np.array(x).T.dot(Q).dot(x)


def draw_mis(G, mis):
    fig = plt.figure()
    # pos = nx.spring_layout(G)
    pos = nx.circular_layout(G)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_nodes(G, pos, nodelist=mis, node_color='blue')
    plt.axis('off')
    plt.show()


def verify_mis(G, mis):
    edge_set = G.edges()
    for v1 in mis:
        for v2 in mis:
            if (v1, v2) in edge_set:
                return False

    return True


def draw_chains():
    G_odd = nx.Graph()
    for i in range(5):
        G_odd.add_node(i)
    for i in range(4):
        G_odd.add_edge(i, i+1)

    edge_labels = {(i, i+1): 2 for i in range(4)}
    labels = {i: r'$-\frac{1}{2}$' if i==0 or i==4 else r'-1' for i in range(5)}
    pos = nx.spectral_layout(G_odd)
    nx.draw(G_odd, pos=pos, node_size=2000, node_color='white')
    nx.draw_networkx_edge_labels(G_odd, pos, edge_labels=edge_labels, font_size=20)
    nx.draw_networkx_labels(G_odd, pos, labels=labels, font_size=20)

    plt.axis("off")
    plt.show()

    G_even = nx.Graph()
    for i in range(4):
        G_even.add_node(i)
    for i in range(3):
        G_even.add_edge(i, i+1)

    pos = nx.spectral_layout(G_even)
    edge_labels = {(i, i+1): 2 if not i==1 else -2 for i in range(3)}
    nx.draw(G_even, pos=pos, node_size=2000, node_color='white')
    labels = {i: r'$-\frac{1}{2}$' if not (i==1 or i==2) else '1' for i in range(4)}
    nx.draw_networkx_edges(G_even, pos, edgelist=[(1, 2)], width=5)
    nx.draw_networkx_labels(G_even, pos, labels=labels, font_size=20)
    nx.draw_networkx_edge_labels(G_even, pos, edge_labels=edge_labels, font_size=20)
    plt.axis("off")
    plt.show()


def write_experiment(problem_ids, result_ids):
    d_exp = {'problem_ids': problem_ids, 'result_ids': result_ids}
    exp_id = find(d_exp, 'experiment')

    return exp_id


def update_record_energies():
    total = db['result'].find().count()
    for i, res in enumerate(db['result'].find()):
        print 'Updating via result {} / {}'.format(i+1, total)
        prob = db['problem'].find_one({'_id': res['problem_id']})
        best_energy = min(res['energies'])

        if best_energy < prob.get('record_energy', np.inf):
            db['problem'].update({'_id': res['problem_id']}, {'$set': {'record_energy': best_energy}})

    print 'Done.'


def wrap_experiments(label, exp_ids):
    wrapper = {'label': label, 'experiment_ids': exp_ids}
    wrapper_id = find(wrapper, 'experiment_wrapper')
    return wrapper_id


def create_result_indexes():
    print 'Starting to create index...'
    db['result'].create_index([('problem_id', ASCENDING), ('solver_name', ASCENDING), ('embedding_id', ASCENDING), ('tuner_id', ASCENDING)])
    # db['result'].ensure_index(['problem_id', 'solver_name', 'embedding_id'])
    print 'Done.'


def update_trial_numbers():
    for i, res in enumerate(db['result'].find()):
        trial = res.get('trial', 0)
        #prob = db['problem'].find_one({'_id': res['problem_id']})

        db['result'].update({'_id': res['_id']}, {'$set': {'trial': trial}})

    print 'Done.'

def update_num_reads():
    for i, res in enumerate(db['result'].find()):
        energies = res.get('energies', [])
        #prob = db['problem'].find_one({'_id': res['problem_id']})

        db['result'].update({'_id': res['_id']}, {'$set': {'num_reads': len(energies)}})

    print 'Done.'


def convert_dimacs_file_to_nx(filename):
    G = nx.Graph()

    with open(filename, 'r') as f:
        dat = f.read()

    lines = dat.splitlines()[1:-1]

    for line in lines:
        _, v1, v2 = line.split()

        G.add_edge(int(v1)-1, int(v2)-1)

    return G


def write_QUBO_to_file(QUBO, filename):

    num_nonzero_diagonals = 0
    max_diagonals = len(QUBO)
    num_off_diagonals = 0
    for i in range(len(QUBO)):
        for j in range(len(QUBO)):
            nonzero = QUBO[i][j] != 0
            if i==j:
                num_nonzero_diagonals += nonzero
            elif i<j:
                num_off_diagonals += nonzero

    with open(filename, 'w') as f:
        s = 'p qubo 0 {} {} {}\n'.format(max_diagonals, num_nonzero_diagonals, num_off_diagonals)
        for i in range(len(QUBO)):
            for j in range(len(QUBO)):
                if QUBO[i][j] != 0:
                    s += '{} {} {}\n'.format(i, j, QUBO[i][j])
        f.write(s)


def best_embedding(prob_id, solver_name, seed, adj, num_attempts):
    smallest = np.inf
    best_emb = None
    for emb_i in range(seed, seed + num_attempts):
        emb = get_embedding(prob_id, solver_name, emb_i, adj)
        num_qubits = sum(map(len, emb['chains']))
        # print emb['chains']
        # print 'embedding {}: {} qubits'.format(emb_i, num_qubits)
        if num_qubits < smallest:
            best_emb = emb
            smallest = num_qubits

    return best_emb


def print_problem_for_cpp(G, filename):
    s = ''
    nodes = G.nodes()
    s += '{}\n'.format(len(nodes))
    for n in nodes:
        set_neigh = set(G.neighbors(n))
        for m in nodes:
            if m in set_neigh:
                s += ' 1 '
            else:
                s += ' 0 '
        s += '\n'
    with open(filename, 'w') as f:
        f.write(s)


def update_tuners_with_chain_strengths():
    for tuner_doc in db['tuner'].find({}):
        result_doc = {'problem_id': tuner_doc['problem_id'], 'tuner_id': tuner_doc['_id'], 'tune': True}
        res = db['result'].find_one(result_doc)
        # print {k: v for k, v in res.iteritems() if '_id' in k or 'strength' in k}
        doc_to_update = {'$set': {'chain_strength': res['chain_strength']}}
        db['tuner'].update({'_id': tuner_doc['_id']}, doc_to_update)
    print 'Done!'

if __name__ == '__main__':
    update_num_reads()
    exit()
    create_result_indexes()
    exit()
    update_tuners_with_chain_strengths()
    exit()
    dimacs = 'graphs/a265032_1dc.64.txt'
    G = convert_dimacs_file_to_nx(dimacs)
    print_problem_for_cpp(G, "graphs/testing.txt")
    exit()

    import os
    # create_result_indexes()
    # convert_dimacs_file_to_nx('graphs/a265032_1dc.64.txt')
    path = 'graphs/frb30-15-mis'
    name='frb30-15-3.mis'
    print 'Converting file to NetworkX graph.'
    # G = convert_dimacs_file_to_nx('graphs/a265032_1dc.2048.txt')
    G = convert_dimacs_file_to_nx('{}/{}'.format(path, name))
    print 'Calling qbsolv.'
    QUBO = generate_mis_qubo(G)
    qubo_file = '{}_qubo.txt'.format(name)
    write_QUBO_to_file(QUBO, qubo_file)
    results_file = '{}_result.txt'
    os.system("/opt/local/qOp/bin/qbsolv -i {} -o {} -n 100".format(qubo_file, results_file))
    print 'qbsolv done.'
    with open(results_file, 'r') as f:
        s = f.read()
        lines = s.splitlines()
    sol = lines[1]
    sol = map(int, list(sol))
    mis = np.flatnonzero(np.array(sol))
    print verify_mis(G, mis)
    print 'MIS size: ', len(mis)
    print mis
    print 'Number of varibales in graph: ', len(sol)
    exit()

    # G = generate_graph(50, 10, 0)
    results = {'D-WAVE WINS': 0, 'TIE': 0, 'NETWORKX WINS': 0, 'D-WAVE LOSES': 0}
    # for seed in range(1):
    for size in range(18, 19):
        print 'Size: ', 4*size
        G = nx.barabasi_albert_graph(4*size, size, 0)

        # G = generate_graph(50, 10, 0)
        dwave_mis = find_max_ind_set_dwave(G, local=False)

        # draw_mis(G, dwave_mis)

        dwave_verified = verify_mis(G, dwave_mis)
        print ''
        print 'Is D-Wave MIS of size {} an IS?'.format(len(dwave_mis)), dwave_verified

        networkx_mis = find_max_ind_set_networkx(G)
        networkx_verified = verify_mis(G, networkx_mis)
        print 'Is NetworkX MIS of size {} an IS?'.format(len(networkx_mis)), networkx_verified
        print ''
