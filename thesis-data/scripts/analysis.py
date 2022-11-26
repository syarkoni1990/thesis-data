__author__ = 'syarkoni'

import matplotlib as mpl
mpl.use('pgf')

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    # "font.size": 8,
    "axes.labelsize": 14,               # LaTeX default is 10pt font.
    "font.size": 16,
    # "legend.title.fontsize": 12,         # Make the legend/label fonts a little smaller
    "legend.fontsize": 10,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    # "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{textcomp}",           # Required for \textmu
        r"\newcommand{\us}{\textmu{}s}",    # Microseconds symbol (\mu s).",
        r"\usepackage{amsmath}",
        r"\usepackage{fixltx2e}",
        ]
    }

import pandas as pd
import seaborn as sns
import numpy as np
import os

import networkx as nx

mpl.rcParams.update(pgf_with_latex)
sns.set_style(pgf_with_latex)

sns.set_style("whitegrid", rc={
    'legend.frameon': True,
    'xtick.major.size': 8.0,
    'xtick.minor.size': 8.0,
    'ytick.major.size': 8.0,
    'ytick.minor.size': 8.0,

    'grid.color': '.1',
    'axes.edgecolor': 'black',
    'ytick.direction': 'in',
    'xtick.direction': 'in',

     'xtick.bottom': True,
     # 'xtick.color': '.15',
     'xtick.top': True,
     # 'ytick.color': '.15',
     'ytick.left': True,
     'ytick.right': True


})
# sns.set_style("ticks")

from pymongo import MongoClient
from bson import ObjectId

client = MongoClient('localhost', 27017)
# db = client['Leiden_MIS']
# db = client['Leiden_MIS_CMA_test']
db = client['Leiden_MIS_tuning_paper']


def is_dwave_solver(solver_name):
    possible_solver_prefixes = ['DW', 'BAY', '2000Q', 'DW2X', 'VFYC']
    state = False
    for pos in possible_solver_prefixes:
        state = state or (pos in solver_name)
    return state


def create_df(exp_id, db):
    exp = db['experiment'].find_one({'_id': exp_id})

    params = ['_id', 'chain_strength', 'embedding_id', 'energies', 'num_occurrences', 'problem_id', 'solutions', 'solver_name', 'trial', 'timing', 'order', 'tuner_id', 'tuner', 'tune']

    merge_over_trials_list = ['energies', 'num_occurrences', 'solutions']
    merge_over_trials = ['_id', 'time_per_sample']

    rows = list(db['result'].find({'_id': {'$in': exp['result_ids']}}, {p: 1 for p in params}))

    _df = pd.DataFrame(rows)

    # for t in ['total_real_time', 'total_init_time', 'total_runtime']:
    for t in ['total_real_time', 'qpu_programming_time', 'run_time_chip']:
        get_timing = lambda row: row['timing'][t]
        _df[t] = _df.apply(get_timing, axis=1)

    # here we should try all the HW stuff and if it doesn't work except and do SW stuff
    try:
        time_per_sample = lambda row: row['timing']['anneal_time_per_run']
        _df['time_per_sample'] = _df.apply(time_per_sample, axis=1)
    except:
        pass

    if 'time_per_sample' not in _df.columns:
        get_time_per_sample = lambda row: float(row['total_runtime']-row['total_init_time']) / sum(row['num_occurrences'])
        _df['time_per_sample'] = _df.apply(get_time_per_sample, axis=1)
    _df = _df.drop('timing', 1)

    time_results = list()
    for solver_name, solv_df in _df.groupby('solver_name'):
        time_res = [{'_id': None}]
        if is_dwave_solver(solver_name):
            # time_res = list(db['result'].find({'_id': {'$in': solv_df['_id'].tolist()}}, {'anneal_time': 1}))
            # for d in time_res:
            #     d['time_per_sample'] = d.pop('time_per_sample', .00002)
            pass
        elif '_sa_' not in solver_name:
            time_res = list(db['result'].find({'_id': {'$in': solv_df['_id'].tolist()}}, {'total_cpu_time': 1}))
            for d in time_res:
                d['time_per_sample'] = d.pop('total_cpu_time')

        time_results.extend(time_res)

    time_df = pd.DataFrame(time_results)
    _df = _df.merge(time_df, on='_id', how='left')
    new_rows = list()

    # Order samples if necessary
    if 'order' in _df.columns:
        for col in ['energies', 'solutions']:
            reorder = lambda row: [row[col][o] for o in row['order']]
            _df[col] = _df.apply(reorder, axis=1)
        new_num_occurrences = lambda row: [1] * len(row['energies'])
        _df['num_occurrences'] = _df.apply(new_num_occurrences, axis=1)

    # Now merge over trials (only for NetworkX)
    if len(_df['trial'].unique()) > 1:
    # if 'chain_strength' not in _df.columns:
        for prob, prob_df in _df.groupby('problem_id'):
            row = dict(prob_df.iloc[0])
            for k in merge_over_trials:
                if type(row[k]) is not list:
                    row[k] = [row[k]]

            for i in range(1, len(prob_df)):
                _row = dict(prob_df.iloc[i])
                for k in merge_over_trials:
                    row[k].extend([_row[k]])
                for k in merge_over_trials_list:
                    row[k].extend(_row[k])
            # print prob_df
            new_rows.append(row)
        exp_df = pd.DataFrame(new_rows)
    else:
        exp_df = _df
        # print exp_df
        # exit()
        _expand_time_per_sample = lambda r: [r['time_per_sample']]*len(r['energies'])
        exp_df['time_per_sample'] = exp_df.apply(_expand_time_per_sample, axis=1)

    exp_df.rename(columns={'_id': 'result_id', 'seed': 'problem_seed'}, inplace=True)

    if 'embedding_id' in _df.columns and exp_df['embedding_id'].count() > 0:
        emb_rows = list()

        for emb_id in exp_df['embedding_id'].unique():
            emb_rows.append(db['embedding'].find_one({'_id': emb_id}))

        emb_df = pd.DataFrame(emb_rows)

        emb_df.drop('problem_id', 1, inplace=True)

        emb_df.rename(columns={'_id': 'embedding_id', 'seed': 'embedding_seed'}, inplace=True)

        exp_df = exp_df.merge(emb_df, on='embedding_id')

    if 'tuner_id' in _df.columns:

        tuner_rows = list()
        for tuner_id in exp_df['tuner_id'].unique():
            if type(tuner_id) == ObjectId:
                tuner_rows.append(db['tuner'].find_one({'_id': tuner_id}, {'sigma': 1, 'with_restarts': 1, 'opt_param': 1, 'opt_fitness': 1, 'param': 1, 'fitness': 1, 'initial_offsets': 1, 'tuner': 1}))

        tuner_df = pd.DataFrame(tuner_rows)
        tuner_df.rename(columns={'_id': 'tuner_id'}, inplace=True)

        exp_df = exp_df.merge(tuner_df, on='tuner_id', how='left')
        get_tune_run = lambda row: '{}_{}_{}'.format(row['tuner'], row['tune'], row['initial_offsets']) if row['tuner'] is not 'None' else 'None'

        exp_df['tuner_run'] = exp_df.apply(get_tune_run, axis=1)


    # This is special for SA. Gonna have to fix this for the general case.
    if False:
        for param in ['beta_start', 'beta_end', 'num_sweeps', 'num_reads']:
            get_param = lambda row: float(row['solver_name'].split(param)[1].split('_')[0])
            _df['solver.params.{}'.format(param)] = _df.apply(get_param, axis=1)
        get_solver_name = lambda row: row['solver_name'].split(':')[0]
        _df['solver_name'] = _df.apply(get_solver_name, axis=1)

    return exp_df


def multi_create_df(exp_label):
    exp_wrap = list(db['experiment_wrapper'].find({'label': exp_label}))[0]
    if not os.path.exists('data/{}/'.format(exp_label)):
        os.mkdir('data/{}/'.format(exp_label))
    for exp_i, exp_id in enumerate(exp_wrap['experiment_ids']):
        filename = 'data/{}/{}_{}.pklz'.format(exp_label, exp_label, exp_i)
        if not os.path.isfile(filename):
            print 'DataFrame {}/{}.'.format(exp_i+1, len(exp_wrap['experiment_ids']))
            exp_df = create_df(exp_id, db)

            exp_df.to_pickle(filename)

    return len(exp_wrap['experiment_ids'])


def multi_munge(basename, num_files):
    for num in range(num_files):
        munged_df = munge('data/{}/{}_{}.pklz'.format(basename, basename, num), db, False)
        munged_df.to_pickle('data/{}/{}_{}_munged.pklz'.format(basename, basename, num))


def calc_MIS_stats_from_row(row, prob):
    new_row = row
    Q = {(i, j): val for [i, j, val] in prob['Q']}
    solutions_interp = map(np.flatnonzero, np.array(row['solutions']))

    def is_ind_set(ind_set):
        for i in ind_set:
            for j in ind_set:
                if Q.get((i, j)) == 2:
                    return False
        return True

    new_row['independent_sets'] = [sol if is_ind_set(sol) else [] for sol in solutions_interp]
    new_row['independent_sets_sizes'] = map(len, new_row['independent_sets'])
    new_row['mis_size'] = max(new_row['independent_sets_sizes'])

    new_row['success_prob'] = sum(np.isclose(new_row['energies'], prob['record_energy']) * new_row['num_occurrences'])/float(sum(new_row['num_occurrences']))
    new_row['tts'] = new_row['success_prob']*sum(new_row['time_per_sample'])
    new_row['distance_from_record'] = (np.array(new_row['energies']) - prob['record_energy']).tolist()
    new_row['closest_to_record'] = min(new_row['distance_from_record'])
    new_row['problem.record_energy'] = prob['record_energy']
    new_row['time_to_closest'] = sum(new_row['time_per_sample'][:np.argmin(new_row['distance_from_record'])+1])

    # Now do embedding stats (if possible):
    if row.get('embedding_id'):
        new_row['chain_lengths'] = map(len, new_row['chains'])
        new_row['num_qubits'] = sum(new_row['chain_lengths'])
        new_row['mean_chain_length'] = np.mean(new_row['chain_lengths'])

    for param in prob['params']:
        new_row['problem.params.{}'.format(param)] = prob['params'][param]

    new_row.pop('solutions')
    new_row.pop('num_occurrences')

    return new_row


def munge(filename, db, save=True):
    df = pd.read_pickle(filename)

    new_rows = list()
    for _, row in df.iterrows():
        prob = db['problem'].find_one({'_id': row['problem_id']})

        new_rows.append(calc_MIS_stats_from_row(row, prob))


    new_df = pd.DataFrame(new_rows)
    if save:
        new_df.to_pickle('munged_{}'.format(filename))
    return new_df


def multi_analyze(basename, num_files, x, y):
    """
    Function that strips out all the extra info from the dataframe and only stores the x and y variables to be plotted.
    This saves a lot of RAM. x and y can be lists.
    :param basename:
    :param num_files:
    :param x:
    :param y:
    :return:
    """
    if type(x) is not list:
        x = [x]
    if type(y) is not list:
        y = [y]

    for num in range(num_files):
        analyzed_df = analyze('data/{}/{}_{}_munged.pklz'.format(basename, basename, num), x, y)
        analyzed_df.to_pickle('data/{}/{}_{}_analyzed.pklz'.format(basename, basename, num))


def analyze(filename, x, y):
    df = pd.read_pickle(filename)
    # print 'Loaded.'

    # from scipy.stats import linregress
    x_cols = set(x)
    y_cols = set(y)
    all_cols = df.columns
    for col in all_cols:
        if col not in x_cols and col not in y_cols:
            df.drop(col, axis=1, inplace=True)


    # new_rows = list()
    # for prob_id, prob_df in df.groupby('problem_id'):
    #     row = dict()
    #
    #     slope, intercept, rval, pval, stderr = linregress(prob_df[x], prob_df[y])
    #
    #     row['slope'] = slope
    #     row['intercept'] = intercept
    #     row['rval'] = rval
    #     row['pval'] = pval
    #     row['stderr'] = stderr
    #     row['mean_chain_length'] = prob_df['mean_chain_length'].mean()
    #
    #     new_rows.append(row)


    # df = pd.DataFrame(new_rows)
    return df

    # fg = sns.FacetGrid(df, hue='problem_id', size=10)
    #
    # fg.map(plt.scatter, x, y, s=60)
    #
    # sns.regplot(x, y, data=df, scatter=False)


def plot(df, x, y, f, **params):
    if type(df) is list:
        df = pd.concat([pd.read_pickle(d) for d in df])

    # print df
    # exit()
    # import matplotlib.pyplot as plt

    # sns.boxplot(y='slope', data=df, orient='v')
    # sns.swarmplot(y='slope', data=df, orient='v', color='black')
    # plt.ylabel('Slope of regression line')
    # plt.xlabel('Aggregated problem instances')
    # fig = plt.gcf()
    #
    # sns.plt.legend(loc=0)
    #
    # plt.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.975, wspace=0, hspace=0)

    # fig.savefig('figures/{}_vs_{}_boxplot.png'.format(y, x))
    figs = f(df, x, y, **params)
    for fig_i, fig in enumerate(figs):
        fig.savefig('figures/{}/{}_vs_{}.pdf'.format(params.get('base', ''), y, x), dpi=800)


def get_best_chain_strength_performance(datafiles):
    dfs = list()
    for dat in datafiles:
        dfs.append(pd.read_pickle(dat))

    df = pd.concat(dfs)
    rows = list()
    for size, n_df in df.groupby('problem.params.n'):
        row = dict()
        row['problem.params.n'] = size
        row['solver_name'] = n_df['solver_name'].iloc[0]

        temp_rows = list()
        for chain_strength in n_df['chain_strength'].unique():
            temp_row = dict()
            temp_row['chain_strength'] = chain_strength
            temp_row['success_prob'] = n_df[n_df['chain_strength'] == chain_strength]['success_prob'].mean()
            temp_row['mean_chain_length'] = n_df[n_df['chain_strength'] == chain_strength]['mean_chain_length'].median()
            temp_row['problem_ids'] = n_df[n_df['chain_strength'] == chain_strength]['problem_id'].tolist()
            temp_row['closest_to_record'] = n_df[n_df['chain_strength'] == chain_strength]['closest_to_record'].tolist()
            temp_row['time_to_closest'] = n_df[n_df['chain_strength'] == chain_strength]['time_to_closest'].tolist()

            temp_rows.append(temp_row)

        temp_df = pd.DataFrame(temp_rows)

        row['chain_length'] = temp_df[temp_df['success_prob'] == temp_df['success_prob'].max()]['mean_chain_length'].min()
        row['chain_strength'] = temp_df[temp_df['success_prob'] == temp_df['success_prob'].max()]['chain_strength'].min()
        row['success_prob'] = temp_df['success_prob'].max()
        row['problem_ids'] = temp_df[temp_df['success_prob'] == temp_df['success_prob'].max()]['problem_ids']
        row['closest_to_record'] = temp_df[temp_df['success_prob'] == temp_df['success_prob'].max()]['closest_to_record']
        row['time_to_closest'] = temp_df[temp_df['success_prob'] == temp_df['success_prob'].max()]['time_to_closest'].tolist()[0]

        rows.append(row)

    new_df = pd.DataFrame(rows)

    more_rows = list()
    for _, row in new_df.iterrows():
        more_row = dict()
        for key in ['chain_length', 'chain_strength', 'problem.params.n', 'solver_name', 'success_prob']:
            if type(row[key]) != list:
                more_row[key] = row[key]

        problem_ids = row['problem_ids'].tolist()[0]
        closest_to_record = row['closest_to_record'].tolist()[0]
        time_to_closest = row['time_to_closest']
        # more_row['time_to_closest'] = row['time_to_closest']

        for i in range(len(problem_ids)):
            more_row['problem_id'] = problem_ids[i]
            more_row['closest_to_record'] = closest_to_record[i]
            more_row['time_to_closest'] = time_to_closest[i]

            more_rows.append(more_row.copy())
    final_df = pd.DataFrame(more_rows)
    final_df.to_pickle('data/random_graphs_p=.2_60/random_graphs_p=.2_60_final.pklz')


def make_success_prob(df, x, y, **params):
    import matplotlib.pyplot as plt

    # fg = sns.FacetGrid(df, hue='problem_id', size=10)

    # fg.map(plt.scatter, x, 'success_prob', s=60)

    # sns.regplot(x, 'success_prob', data=df, scatter=False)

    semilogy = params.pop('semilogy', False)

    alpha = params.pop('alpha', 1.)
    fg = sns.pointplot(x, 'success_prob', data=df, estimator=np.mean, **params)
    for col in fg.collections:
        col.set_alpha(alpha)
    for lin in fg.lines:
        lin.set_alpha(alpha)

    # if 'chain_strength' in df.columns:
    #     rows = list()
    #     for size, n_df in df.groupby('problem.params.n'):
    #         row = dict()
    #         row['problem.params.n'] = size
    #         row['solver_name'] = n_df['solver_name'].iloc[0]
    #
    #         temp_rows = list()
    #         for chain_strength in n_df['chain_strength'].unique():
    #             temp_row = dict()
    #             temp_row['chain_strength'] = chain_strength
    #             temp_row['success_prob'] = n_df[n_df['chain_strength'] == chain_strength]['success_prob'].mean()
    #             temp_row['mean_chain_length'] = n_df[n_df['chain_strength'] == chain_strength]['mean_chain_length'].median()
    #             temp_row['problem_ids'] = n_df[n_df['chain_strength'] == chain_strength]['problem_id'].tolist()
    #             temp_row['closest_to_record'] = n_df[n_df['chain_strength'] == chain_strength]['closest_to_record'].tolist()
    #
    #             temp_rows.append(temp_row)
    #
    #         temp_df = pd.DataFrame(temp_rows)
    #
    #         row['chain_length'] = temp_df[temp_df['success_prob'] == temp_df['success_prob'].max()]['mean_chain_length'].min()
    #         row['chain_strength'] = temp_df[temp_df['success_prob'] == temp_df['success_prob'].max()]['chain_strength'].min()
    #         row['success_prob'] = temp_df['success_prob'].max()
    #         row['problem_ids'] = temp_df[temp_df['success_prob'] == temp_df['success_prob'].max()]['problem_ids']
    #         row['closest_to_record'] = temp_df[temp_df['success_prob'] == temp_df['success_prob'].max()]['closest_to_record']
    #
    #         rows.append(row)
    #
    #     new_df = pd.DataFrame(rows)
    #     more_rows = list()
    #     for _, row in new_df.iterrows():
    #         more_row = dict()
    #         for key in ['chain_length', 'chain_strength', 'problem.params.n', 'solver_name', 'success_prob']:
    #             if type(row[key]) != list:
    #                 more_row[key] = row[key]
    #
    #         problem_ids = row['problem_ids'].tolist()[0]
    #         closest_to_record = row['closest_to_record'].tolist()[0]
    #         for i in range(len(problem_ids)):
    #             more_row['problem_id'] = problem_ids[i]
    #             more_row['closest_to_record'] = closest_to_record[i]
    #
    #             more_rows.append(more_row.copy())
    # else:
    #     new_df = df

    # print more_rows
    # final_df = pd.DataFrame(more_rows)
    # final_df.to_pickle('data/random_graphs_p=.2_60/random_graphs_p=.2_60_final.pklz')
    # exit()
    sns.pointplot(x=df['n'], y=df['success_prob'], color='black', markers=['x'])

    fig = plt.gcf()

    sns.plt.legend(loc=0)

    plt.ylabel('Success probability')
    plt.xlabel(x)

    if semilogy:
        plt.semilogy()
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.975, wspace=0, hspace=0)

    # fig.savefig('{}_vs_{}_TEST.png'.format('success_prob', x))
    return [fig]


def multi_analysis(exp_wrapper_label, x_cols, y_cols, create_dfs=True, munge_dfs=True, analyze_dfs=True):
    """
    NOT THE SAME AS 'MULTI_ANALYZE'
    :param exp_wrapper_label:
    :return:
    """

    directories = ['data/{}/'.format(exp_wrapper_label), 'figures/{}/'.format(exp_wrapper_label)]
    for dir in directories:
        if not os.path.exists(dir):
            os.makedirs(dir)

    exp_wrap = db['experiment_wrapper'].find_one({'label': exp_wrapper_label})
    num_experiments = len(exp_wrap['experiment_ids'])
    if create_dfs:
        print 'Creating dataframes...'
        multi_create_df(exp_wrapper_label)

    if munge_dfs:
        print 'Munging data...'
        multi_munge(exp_wrapper_label, num_experiments)

    if analyze_dfs:
        print 'Analyzing data...'
        multi_analyze(exp_wrapper_label, num_experiments, x_cols, y_cols)

    print 'Done.'


def multi_plot(exp_wrapper_label, x, y, f, **params):
    params['base'] = params.get('base', exp_wrapper_label)
    if not os.path.exists('figures/{}'.format(params['base'])):
        os.makedirs('figures/{}'.format(params['base']))

    if type(exp_wrapper_label) == list:
        datafiles = list()
        for exp_wrap_label in exp_wrapper_label:
            exp_wrap = db['experiment_wrapper'].find_one({'label': exp_wrap_label})
            for i in range(len(exp_wrap['experiment_ids'])):
                datafiles.append('data/{}/{}_{}_analyzed.pklz'.format(exp_wrap_label, exp_wrap_label, i))
    else:
        exp_wrap = db['experiment_wrapper'].find_one({'label': exp_wrapper_label})
        datafiles = ['data/{}/{}_{}_analyzed.pklz'.format(exp_wrapper_label, exp_wrapper_label, i) for i in range(len(exp_wrap['experiment_ids']))]

    # datafiles.append('data/random_graphs_p=.2_60/random_graphs_p=.2_60_final.pklz')
    plot(datafiles, x, y, f, **params)


def make_multiembed_boxplot(df, x, y, **params):
    import matplotlib.pyplot as plt
    from scipy.stats import linregress

    x = 'mean_chain_length'
    y = 'closest_to_record'

    new_rows = list()
    for prob_id, prob_df in df.groupby('problem_id'):
        row = dict()

        slope, intercept, rval, pval, stderr = linregress(prob_df[x], prob_df[y])

        row['slope'] = slope
        row['intercept'] = intercept
        row['rval'] = rval
        row['pval'] = pval
        row['stderr'] = stderr
        row['mean_chain_length'] = prob_df['mean_chain_length'].mean()

        new_rows.append(row)

    df = pd.DataFrame(new_rows)
    sns.boxplot(y='slope', data=df, orient='v', width=.5, linewidth=0.75)
    # plt.boxplot(df['slope'])
    sns.swarmplot(y='slope', data=df, orient='v', color='black')
    plt.plot([], [], 'o', label='Linear regression data', color='black')

    fig = plt.gcf()

    sns.plt.legend(loc=0)
    return [fig]


def make_mapped_boxplot(df, x, y, **params):
    import matplotlib.pyplot as plt
    params.pop('base', '')
    # _get_solver_name = lambda row: row if row else 'NetworkX'
    # df['solver'].fillna('NetworkX', inplace=True)
    # print len(df)
    # exit()
    sns.boxplot(x=x, y=y, data=df, orient='v', width=.5, linewidth=0.75, **params)

    fig = plt.gcf()

    # sns.plt.legend(loc=0)

    plt.ylabel(y)
    plt.xlabel(x)

    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.975, wspace=0, hspace=0)

    # fig.savefig('{}_vs_{}_TEST.png'.format('success_prob', x))
    return [fig]


def pointplot(df, x, y, **params):
    df = df[(df['tuner_run'] == 'cma_False_uniform') | (df['tuner_run'] == 'cma_False_None') | (df['tuner_run'] == 'nan_nan_nan')]

    semilogy = params.pop('semilogy', False)
    # df[df['solver_name'] == 'BAY1_2000Q']['time_to_closest'] *= .000001
    # df['solver.params.num_sweeps'] = df['solver.params.num_sweeps'].fillna(0)
    # df = df[df['solver.params.num_sweeps'] < 11]

    # this converts time from us to s
    # _correct_times = lambda row: row['time_to_closest']*.000001 if row['solver_name'] == 'BAY1_2000Q' or '_sa_' in row['solver_name'] else row['time_to_closest']
    _correct_times = lambda row: row['time_to_closest']*.000001 if row['solver_name'] == 'DW_2000Q_2' or '_sa_' in row['solver_name'] else row['time_to_closest']
    df['time_to_closest'] = df.apply(_correct_times, axis=1)

    good_df = df.ix[~(df['success_prob'] == 0.)]
    tuner_names = {'cma_False_uniform': 'CMA (initial = uniform)', 'cma_False_None': 'CMA (initial = 0)', 'nan_nan_nan': 'No tuning', 'None': 'No tuning'}

    _fix_tuner_names = lambda row: tuner_names[row['tuner_run']]


    good_df['Tuning method'] = good_df.apply(_fix_tuner_names, axis=1)
    if params.get('hue') == 'tuner_run': params['hue'] = 'Tuning method'
    # bad_df = df.ix[(df['success_prob'] == 0.)]
    # bad_df['success_prob'] = .00001

    # new_rows = list()
    # for (tuner_run, n), _df in df.groupby(['tuner_run', 'problem.params.n']):
    #     # row = {'tuner_run': tuner_run, 'problem.params.n': n, 'num_solved': _df.ix[~(_df['success_prob'] == 0)]['success_prob'].count(), 'num_unsolved': _df.ix[(_df['success_prob'] == 0)]['success_prob'].count()}
    #     row = {'tuner_run': tuner_run, 'problem.params.n': n, 'success_prob_median': _df.ix[~(_df['success_prob'] == 0)]['success_prob'].median(), 'success_prob_mean': _df.ix[~(_df['success_prob'] == 0)]['success_prob'].mean()}
    #     row = {'tuner_run': tuner_run, 'problem.params.n': n, 'success_prob_median': _df['success_prob'].median(), 'success_prob_mean': _df['success_prob'].mean()}
    #     new_rows.append(row)
    # new_df = pd.DataFrame(new_rows)

    # new_new_rows = list()
    # for n, _df in new_df.groupby('problem.params.n'):
    #     for tuner in ['cma_False_None', 'cma_False_uniform']:
    #         row = {'tuner_run': tuner, 'fraction_improvement': float(_df.ix[(_df['tuner_run'] == tuner)]['success_prob_mean'])/float(_df.ix[(_df['tuner_run'] == 'nan_nan_nan')]['success_prob_mean']), 'problem.params.n': n}
    #         new_new_rows.append(row)
    #
    # new_new_df = pd.DataFrame(new_new_rows)


    # sns.pointplot(x=x, y='fraction_improvement', data=new_df, estimator=np.mean, dodge=.3, scatter_kws={"ms": 1}, **params)
    # sns.pointplot(x=x, y='fraction_improvement', data=new_new_df, estimator=np.mean, dodge=.3, scatter_kws={"ms": 1}, **params)
    sns.pointplot(x=x, y=y, data=good_df, estimator=np.mean, dodge=.3, scatter_kws={"ms": 1}, **params)


    # Use JointGrid directly to draw a custom plot
    # grid = sns.JointGrid(data=good_df, x=x, y=y)
    # grid.plot_joint(sns.pointplot, hue='tuner_run')
    # grid.plot_marginals(sns.boxplot, hue='tuner_run')

    # p = sns.JointGrid(x=df['x'],y=df['y'],)
    #
    # p = p.plot_joint(plt.scatter)
    #
    # p.ax_marg_x.hist(df['x'],alpha = 0.5)
    #
    # p.ax_marg_y.hist(df['y'],orientation = 'horizontal',alpha = 0.5)
    #
    # p.ax_marg_x.hist(df['z'],alpha = 0.5,range = (np.min(df['x']), np.max(df['x'])))


    import matplotlib.pyplot as plt
    fig = plt.gcf()

    # sns.plt.legend(loc=0)
    if semilogy:
        plt.semilogy()

    plt.ylabel('Success probability')
    plt.xlabel('Problem size')

    plt.ylim([.00001, 1])

    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.975, wspace=0, hspace=0)

    # fig.savefig('{}_vs_{}_TEST.png'.format('success_prob', x))
    return [fig]


def fraction_improvement(df, x, y, **params):
    semilogy = params.pop('semilogy', False)
    # df[df['solver_name'] == 'BAY1_2000Q']['time_to_closest'] *= .000001
    # df['solver.params.num_sweeps'] = df['solver.params.num_sweeps'].fillna(0)
    # df = df[df['solver.params.num_sweeps'] < 11]

    # this converts time from us to s
    # _correct_times = lambda row: row['time_to_closest']*.000001 if row['solver_name'] == 'BAY1_2000Q' or '_sa_' in row['solver_name'] else row['time_to_closest']
    _correct_times = lambda row: row['time_to_closest']*.000001 if row['solver_name'] == 'DW_2000Q_2' or '_sa_' in row['solver_name'] else row['time_to_closest']
    df['time_to_closest'] = df.apply(_correct_times, axis=1)

    df = df.ix[~(df['success_prob'] == 0.)]

    # bad_df = df.ix[(df['success_prob'] == 0.)]
    # bad_df['success_prob'] = .00001
    num_bootstrap = 10

    new_rows = list()
    for (tuner_run, n), _df in df.groupby(['tuner_run', 'problem.params.n']):
        successes = _df['success_prob'].tolist()
        num_successes = len(successes)
        for boot_i in range(num_bootstrap):
            boot_successes = np.random.choice(successes, num_successes)
            row = {'tuner_run': tuner_run, 'problem.params.n': n,
                   'success_prob_median': np.median(boot_successes),
                   'success_prob_mean': np.mean(boot_successes),
                   'boot_i': boot_i,

            }
            new_rows.append(row)

        # row = {'tuner_run': tuner_run, 'problem.params.n': n, 'num_solved': _df.ix[~(_df['success_prob'] == 0)]['success_prob'].count(), 'num_unsolved': _df.ix[(_df['success_prob'] == 0)]['success_prob'].count()}
        # row = {'tuner_run': tuner_run, 'problem.params.n': n, 'success_prob_median': _df.ix[~(_df['success_prob'] == 0)]['success_prob'].median(), 'success_prob_mean': _df.ix[~(_df['success_prob'] == 0)]['success_prob'].mean()}
        # row = {'tuner_run': tuner_run, 'problem.params.n': n, 'success_prob_median': _df['success_prob'].median(), 'success_prob_mean': _df['success_prob'].mean(), 'success_prob_25': np.percentile(_df['success_prob'], 25), 'success_prob_75': np.percentile(_df['success_prob'], 75)}
    new_df = pd.DataFrame(new_rows)
    # print new_df
    # exit()
    new_new_rows = list()
    for n, _df in new_df.groupby('problem.params.n'):
        print n
        # print _df
        # exit()
        for tuner in ['cma_False_None', 'cma_False_uniform']:
            for boot_i in range(num_bootstrap):
                for boot_j in range(num_bootstrap):
                    row = {'tuner_run': tuner,
                           'boot_i': boot_i,
                           'fraction_improvement': float(_df.ix[(_df['tuner_run'] == tuner) & (_df['boot_i'] == boot_i)]['success_prob_mean'])/float(_df.ix[(_df['tuner_run'] == 'nan_nan_nan') & (_df['boot_i'] == boot_j)]['success_prob_mean']), 'problem.params.n': n}
                    new_new_rows.append(row)
            # row = {'tuner_run': tuner, 'fraction_improvement': float(_df.ix[(_df['tuner_run'] == tuner)]['success_prob_mean'])/float(_df.ix[(_df['tuner_run'] == 'nan_nan_nan')]['success_prob_mean']), 'problem.params.n': n}

    new_new_df = pd.DataFrame(new_new_rows)

    tuner_names = {'cma_False_uniform': 'CMA (initial = uniform)', 'cma_False_None': 'CMA (initial = 0)', 'nan_nan_nan': 'No tuning', 'None': 'No tuning'}

    _fix_tuner_names = lambda row: tuner_names[row['tuner_run']]


    new_new_df['Tuning method'] = new_new_df.apply(_fix_tuner_names, axis=1)
    if params.get('hue') == 'tuner_run': params['hue'] = 'Tuning method'

    for (n, tuning_method), _df in new_new_df.groupby(['problem.params.n', 'Tuning method']):
        print n, tuning_method, _df['fraction_improvement'].mean()

    # sns.pointplot(x=x, y='fraction_improvement', data=new_df, estimator=np.mean, dodge=.3, scatter_kws={"ms": 1}, **params)
    sns.pointplot(x=x, y='fraction_improvement', data=new_new_df, dodge=.2, scatter_kws={"ms": 1}, errwidth=3,caps=2,**params)
    # sns.pointplot(x=x, y=y, data=good_df, estimator=np.mean, dodge=.3, scatter_kws={"ms": 1}, **params)


    # Use JointGrid directly to draw a custom plot
    # grid = sns.JointGrid(data=good_df, x=x, y=y)
    # grid.plot_joint(sns.pointplot, hue='tuner_run')
    # grid.plot_marginals(sns.boxplot, hue='tuner_run')

    # p = sns.JointGrid(x=df['x'],y=df['y'],)
    #
    # p = p.plot_joint(plt.scatter)
    #
    # p.ax_marg_x.hist(df['x'],alpha = 0.5)
    #
    # p.ax_marg_y.hist(df['y'],orientation = 'horizontal',alpha = 0.5)
    #
    # p.ax_marg_x.hist(df['z'],alpha = 0.5,range = (np.min(df['x']), np.max(df['x'])))


    import matplotlib.pyplot as plt
    fig = plt.gcf()

    # sns.plt.legend(loc=0)
    if semilogy:
        plt.semilogy()

    plt.ylabel('Improvement ratio')
    plt.xlabel('Problem size')

    plt.ylim([.1, 30])

    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.975, wspace=0, hspace=0)

    # fig.savefig('{}_vs_{}_TEST.png'.format('success_prob', x))
    return [fig]


def boxplot(df, x, y, **params):
    df = df[(df['tuner_run'] == 'cma_False_uniform') | (df['tuner_run'] == 'cma_False_None') | (df['tuner_run'] == 'nan_nan_nan')]


    semilogy = params.pop('semilogy', False)
    params.pop('base')
    # df[df['solver_name'] == 'BAY1_2000Q']['time_to_closest'] *= .000001
    # df['solver.params.num_sweeps'] = df['solver.params.num_sweeps'].fillna(0)
    # df = df[df['solver.params.num_sweeps'] < 11]

    # this converts time from us to s
    # _correct_times = lambda row: row['time_to_closest']*.000001 if row['solver_name'] == 'BAY1_2000Q' or '_sa_' in row['solver_name'] else row['time_to_closest']
    _correct_times = lambda row: row['time_to_closest']*.000001 if row['solver_name'] == 'DW_2000Q_1' or '_sa_' in row['solver_name'] else row['time_to_closest']
    df['time_to_closest'] = df.apply(_correct_times, axis=1)
    # _correct_tuner_run = lambda row: 'None' if str(row['tuner_run']) == 'nan_nan_nan' else str(row['tuner_run'])
    # df['tuner_run'] = df.apply(_correct_tuner_run, axis=1)

    tuner_names = {'cma_False_uniform': 'CMA (initial = uniform)', 'cma_False_None': 'CMA (initial = 0)', 'nan_nan_nan': 'No tuning', 'None': 'No tuning'}

    _fix_tuner_names = lambda row: tuner_names[row['tuner_run']]


    df['Tuning method'] = df.apply(_fix_tuner_names, axis=1)
    if params.get('hue') == 'tuner_run': params['hue'] = 'Tuning method'

    new_rows = list()
    for (tuner_run, n), _df in df.groupby(['Tuning method', 'problem.params.n']):
        row = {'Tuning method': tuner_run, 'problem.params.n': n, 'num_solved': _df.ix[~(_df['success_prob'] == 0)]['success_prob'].count(), 'num_unsolved': _df.ix[(_df['success_prob'] == 0)]['success_prob'].count()}
        new_rows.append(row)
    new_df = pd.DataFrame(new_rows)


    sns.barplot(x='problem.params.n', y='num_unsolved', data=new_df, hue='Tuning method',estimator=np.median)#,hue_order=['cma_True_None', 'cma_False_None', 'cma_True_uniform', 'cma_False_uniform', 'None'])

    # sns.boxplot(x=x, y=y, data=new_df, **params)

    import matplotlib.pyplot as plt
    fig = plt.gcf()

    # sns.plt.legend(loc=0)
    # if semilogy:
    #     plt.semilogy()

    plt.ylabel('Number of unsolved instances')
    plt.xlabel('Problem size')
    plt.ylim([0, 50])

    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.975, wspace=0, hspace=0)

    return [fig]


def compare(df, x, y, **params):
    semilogy = params.pop('semilogy', False)
    semilogx = params.pop('semilogx', False)
    params.pop('base')

    # df = pd.DataFrame(df[df['tuner_run'] == 'cma_False' or df['tuner_run'] == 'None'])

    # df[df['solver_name'] == 'BAY1_2000Q']['time_to_closest'] *= .000001
    # df['solver.params.num_sweeps'] = df['solver.params.num_sweeps'].fillna(0)
    # df = df[df['solver.params.num_sweeps'] < 11]

    # this converts time from us to s
    # _correct_times = lambda row: row['time_to_closest']*.000001 if row['solver_name'] == 'BAY1_2000Q' or '_sa_' in row['solver_name'] else row['time_to_closest']
    _correct_times = lambda row: row['time_to_closest']*.000001 if row['solver_name'] == 'DW_2000Q_1' or '_sa_' in row['solver_name'] else row['time_to_closest']
    df['time_to_closest'] = df.apply(_correct_times, axis=1)


    # sns.pointplot(x=x, y=y, data=df, estimator=np.mean, **params)
    sns.regplot(x=df[df['tuner_run'] == 'cma_False']['success_prob'], y=df[df['tuner_run'] == 'None']['success_prob'], fit_reg=False)
    # cma_False = list()
    # cma_None = list()
    #
    # for prob_id in df['problem_id'].unique():
    #     print prob_id
    #     cma_False.append(df[df['tuner_run'] == 'cma_False' and df['problem_id'] == prob_id]['success_prob'])
    #     cma_None.append(df[df['tuner_run'] == 'None' and df['problem_id'] == prob_id]['success_prob'])
    #     print cma_False, cma_None
    #     exit()





    import matplotlib.pyplot as plt
    fig = plt.gcf()

    # sns.plt.legend(loc=0)
    if semilogy:
        plt.semilogy()
    if semilogx:
        plt.semilogx()

    plt.ylabel(y)
    plt.xlabel(x)

    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.975, wspace=0, hspace=0)

    return [fig]

def plot_cma(filename, key=None):
    import json
    import matplotlib.pyplot as plt

    fig = plt.figure()

    with open(filename, 'r') as f:
        data = json.load(f)

    if not key:
        print data.keys()
        return None

    _data = np.array(data[key])

    if len(_data.shape) == 1:
        n = _data.shape[0]
        plt.plot(_data)
    else:
        num_iters, n, _ = _data.shape

        for i in range(n):
            plt.plot(_data[:, i, :])

    fig.savefig(filename.split('.')[0]+'_{}.png'.format(key))

def tuning_improvement(df, x, y, **params):

    semilogy = params.pop('semilogy', False)
    params.pop('base')
    # df[df['solver_name'] == 'BAY1_2000Q']['time_to_closest'] *= .000001
    # df['solver.params.num_sweeps'] = df['solver.params.num_sweeps'].fillna(0)
    # df = df[df['solver.params.num_sweeps'] < 11]

    # this converts time from us to s
    # _correct_times = lambda row: row['time_to_closest']*.000001 if row['solver_name'] == 'BAY1_2000Q' or '_sa_' in row['solver_name'] else row['time_to_closest']
    _correct_times = lambda row: row['time_to_closest']*.000001 if row['solver_name'] == 'DW_2000Q_1' or '_sa_' in row['solver_name'] else row['time_to_closest']
    df['time_to_closest'] = df.apply(_correct_times, axis=1)
    _correct_tuner_run = lambda row: 'None' if str(row['tuner_run']) == 'nan_nan_nan' else str(row['tuner_run'])
    df['tuner_run'] = df.apply(_correct_tuner_run, axis=1)

    # print df['tuner_run'].unique()
    # print float(df[(df.problem_id == ObjectId("5b1c1b98c80ee2268a5ebec1")) & (df.tuner_run == 'None')]['success_prob'])
    # exit()

    _get_tuning_improvement = lambda row: row['success_prob']/float(df[(df.problem_id == row['problem_id']) & (df.tuner_run == 'None')]['success_prob']) if float(df[(df.problem_id == row['problem_id']) & (df.tuner_run == 'None')]['success_prob'])>0 else row['success_prob']
    df['tuning_improvement'] = df.apply(_get_tuning_improvement, axis=1)

    _fix_for_log = lambda row: row['tuning_improvement'] if row['tuning_improvement'] > 0. else 9000 if row['tuner_run'] != 'None' else 1.
    df['tuning_improvement'] = df.apply(_fix_for_log, axis=1)
    # sns.boxplot(x=x, y=y, data=df, **params)
    # sns.pointplot(x=x, y=y, data=df, estimator=np.median, dodge=True, **params)
    sns.lmplot(x=x, y=y, data=df, fit_reg=False,x_jitter=True,scatter_kws={'s':10, 'alpha': .8}, legend=False, **params)

    import matplotlib.pyplot as plt
    fig = plt.gcf()

    # sns.plt.legend(loc=0)
    if semilogy:
        plt.semilogy()

    plt.ylabel(y)
    plt.xlabel(x)
    plt.ylim([0.0001, 10000])
    plt.xlim([10, 50])
    plt.legend(loc=0)
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.975, wspace=0, hspace=0)

    return [fig]


def analyze_experiment(num_problems):
    # label = 'random_graphs_p=.2_CMA_num_problems={}_10k_tune_all_configs'.format(num_problems)
    label = 'random_graphs_p=.2_CMA_num_problems=[20, 25, 30, 35, 40, 45, 50, 55, 60]_10k_tune_all_configs'
    #
    # num_dfs = multi_create_df(label)
    # multi_munge(label, num_dfs)
    # multi_analysis(label, ['tts', 'solver_name', 'problem_id', 'problem.params.p', 'problem.params.n', 'closest_to_record', 'time_per_sample', 'tuner_run', 'chains', 'opt_param'], ['success_prob', 'time_to_closest'])

    # multi_plot(label, 'problem.params.n', 'success_prob', boxplot, hue='tuner_run', semilogy=False)
    # multi_plot(label, 'problem.params.n', 'success_prob', pointplot, hue='tuner_run', semilogy=True)
    multi_plot(label, 'problem.params.n', 'success_prob', fraction_improvement, hue='tuner_run', semilogy=True)
    # multi_plot(label, 'problem.params.n', 'success_prob', offset_vs_chain_length, hue='tuner_run', semilogy=False)
    # multi_plot(label, 'problem.params.n', 'tuning_improvement', tuning_improvement, hue='tuner_run', semilogy=True)
    # multi_plot(label, 'cma_False', 'None', compare)


def offset_vs_chain_length(df, x, y, **params):
    # df = df[(df['tuner_run'] == 'cma_False_uniform') | (df['tuner_run'] == 'cma_False_None')]
    df = df[(df['tuner_run'] == 'cma_False_None')]

    _get_chain_lengths = lambda row: map(len, row['chains'])
    df['chain_lengths'] = df.apply(_get_chain_lengths, axis=1)

    df.fillna(0, inplace=True)

    # _get_last_opt_param = lambda row: map(lambda z: z[-1], row['opt_param']) if row['opt_param'] != 0 else None
    _get_last_opt_param = lambda row: list(row['opt_param'])[-1] if row['opt_param']!=0 else None
    df['final_opt_param'] = df.apply(_get_last_opt_param, axis=1)

    new_rows = list()

    for _, row in df.iterrows():
        prob = db['problem'].find_one({'_id': ObjectId(row['problem_id'])})
        G = nx.Graph()

        for [i, j, k] in prob['Q']:
            if k == -1:
                G.add_node(i)
            else:
                G.add_edge(i, j)

        deg = {node: degree for node, degree in G.degree}

        for c_i, c in enumerate(row['chains']):
            new_rows.append( {'degree': deg[c_i], 'problem.params.n': row['problem.params.n'], 'chain_length': len(c), 'offset': row['final_opt_param'][c_i][0], 'tuner_run': row['tuner_run'], 'success_prob': row['success_prob'], 'closest_to_record': row['closest_to_record']} )

    new_df = pd.DataFrame(new_rows)
    new_df.rename(columns={'problem.params.n': 'Problem size'}, inplace=True)
    # print new_df
    # exit()
    # sns.pointplot(data=new_df,x='chain_length', y='offset',hue='tuner_run')
    # sns.pointplot(data=new_df,x='degree', y='offset',hue='problem.params.n')
    sns.pointplot(data=new_df,x='degree', y='offset',hue='Problem size')
    # sns.lmplot(data=new_df,x='chain_length', y='offset',hue='tuner_run',fit_reg=False,col='problem.params.n', x_jitter=.1)
    # sns.lmplot(data=new_df,x='degree', y='offset',hue='problem.params.n',fit_reg=False, x_jitter=.1)
    import matplotlib.pyplot as plt
    fig = plt.gcf()

    # sns.plt.legend(loc=0)
    # if semilogy:
    #     plt.semilogy()

    plt.ylabel('Offset value')
    plt.xlabel('Node degree')
    plt.ylim([-0.15, .25])
    # plt.xlim([0, 25])

    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.975, wspace=0, hspace=0)

    return [fig]


def cma_training():
    import matplotlib.pyplot as plt

    tuner_id = ObjectId("5b4f3315c80ee24538421dde")
    problem_id = ObjectId("5b4f328dc80ee24538421dd3")
    prob = db['problem'].find_one({'_id': problem_id})
    tuner = db['tuner'].find_one({'_id': tuner_id})
    fitness = tuner['fitness']
    opt_fitness = tuner['opt_fitness']

    # this plots fitness function:
    plt.plot(range(3, 3+len(fitness)), fitness, '--r', label='Fitness function')
    plt.plot(range(3, 3+len(fitness)), opt_fitness, '-b', label='Optimal fitness')
    plt.xlabel('CMA-ES iteration number')
    plt.ylabel('Mean energy')
    plt.ylim([-11, -6])
    plt.xlim([0, 100])
    plt.minorticks_on()
    plt.legend(loc=0)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.975, top=0.975, wspace=0, hspace=0)
    plt.savefig('cma_fitness_tuning.pdf')

    # this plots offset values
    # _data = np.array(tuner['opt_param'])
    # num_iters, n, _ = _data.shape
    #
    # for i in range(n):
    #     plt.plot(range(3, 3+len(_data[:, i, :])), _data[:, i, :])
    #
    # plt.plot([],[],color='black', label='Optimal offset values')
    # plt.xlabel('CMA-ES iteration number')
    # plt.ylabel('Anneal offset value $\Delta s$')
    # plt.xlim([0, 100])
    # plt.legend(loc=0)
    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.975, top=0.975, wspace=0, hspace=0)
    # plt.minorticks_on()
    # plt.savefig('cma_offsets_tuning.pdf')

if __name__ == '__main__':

    cma_training()
    exit()
    analyze_experiment([20, 25, 30, 35, 40, 45, 50, 55, 60])
    exit()

    num_dfs = multi_create_df('random_graphs_p=.2_50_CMA')
    multi_munge('random_graphs_p=.2_50_CMA', num_dfs)
    multi_analysis('random_graphs_p=.2_50_CMA', ['tts', 'solver_name', 'problem_id', 'problem.params.p', 'problem.params.n', 'closest_to_record', 'time_per_sample', 'tuner_run'], ['success_prob', 'time_to_closest'])
    multi_plot('random_graphs_p=.2_50_CMA', 'problem.params.n', 'success_prob', pointplot, hue='tuner_run', semilogy=True)
    exit()

    plot_cma('cma_es_test_chain_n=50_50_p2_min_1000samples.txt', 'A')
    # plot_cma('cma_es_test_chain_n=50_50_p2_min_1000samples.txt', 'opt_fitness')
    exit()
    # multi_analysis('random_graphs_p=.2_60_SA', ['tts', 'solver_name', 'problem_id', 'problem.params.p', 'problem.params.n', 'closest_to_record', 'time_per_sample', 'solver.params.beta_start', 'solver.params.beta_end', 'solver.params.num_reads', 'solver.params.num_sweeps'], ['success_prob', 'time_to_closest'])

    # get_best_chain_strength_performance(['data/random_graphs_p=.2_60/random_graphs_p=.2_60_{}_analyzed.pklz'.format(i) for i in range(450)])
    # exit()
    # multi_plot(['random_graphs_p=.2_60_networkx'], 'problem.params.n', 'time_to_closest', pointplot, hue='solver_name', base='final_combined', semilogy=True)
    # multi_plot('random_graphs_p=.2_60_SA', 'problem.params.n', 'tts', pointplot, hue='solver.params.num_sweeps', semilogy=True)
    # multi_plot(['random_graphs_p=.2_60_SA', 'random_graphs_p=.2_60_networkx'], 'problem.params.n', 'time_to_closest', pointplot, hue='solver_name', base='final_combined', semilogy=True)
    multi_plot(['random_graphs_p=.2_60_SA', 'random_graphs_p=.2_60_networkx'], 'problem.params.n', 'closest_to_record', pointplot, hue='solver_name', base='final_combined', semilogy=False)
    exit()
    # multi_analysis('random_graphs_p=.2_60', ['solver_name', 'problem_id', 'problem.params.p', 'problem.params.n', 'mean_chain_length', 'embedding_seed', 'closest_to_record', 'chain_strength', 'time_per_sample'], ['success_prob', 'time_to_closest'])
    # exit()
    # multi_plot('random_graphs_p=.2_60', 'problem.params.n', 'none', make_success_prob, hue='chain_strength', semilogy=True)
    # multi_plot('random_graphs_p=.2_60', 'problem.params.n', 'mean_chain_length', make_mapped_boxplot)

    # multi_analysis('random_graphs_p=.2_60_networkx', ['solver_name', 'problem_id', 'problem.params.p', 'problem.params.n', 'closest_to_record', 'time_per_sample'], ['success_prob', 'time_to_closest'])
    # multi_plot('random_graphs_p=.2_60', 'problem.params.n', 'wat', make_success_prob, semilogy=True, alpha=0.5)
    # exit()
    # multi_plot('random_graphs_p=.2_60_networkx', 'problem.params.n', 'closest_to_record', make_mapped_boxplot, semilogy=True)


    # multi_plot(['random_graphs_p=.2_60_networkx'], 'problem.params.n', 'closest_to_record', make_mapped_boxplot, hue='solver_name', base='final_combined')
    # multi_plot('random_graphs_p=.2_60', 'problem.params.n', 'closest_to_record', make_mapped_boxplot, semilogy=True, hue='solver')
    # exit()
    # multi_analysis('multiembed_gnp_.2', ['problem_id', 'problem.params.p', 'mean_chain_length', 'embedding_seed', 'closest_to_record'], 'success_prob', create_dfs=False, munge_dfs=False)
    # multi_plot('multiembed_gnp_.2', 'multiembed', 'wat2', make_multiembed_boxplot)
    # exit()
    # multi_create_df('test_gnp_hardness_.425')
    # multi_munge('test_gnp_hardness_.425', 16)
    # multi_analyze('test_gnp_hardness_.425', 16, ['problem.params.p', 'mean_chain_length'], 'success_prob')
    # datafiles = ['data/test_gnp_hardness_.3_{}_munged.pklz'.format(i) for i in range(10)]
    # plot(datafiles, 'problem.params.p', 'success_prob', make_success_prob, semilogy=True, base='test_gnp_hardness_.425')
    # exit()
    # datafiles = ['data/test_gnp_hardness_.425_{}_analyzed.pklz'.format(i) for i in range(16)]
    # plot(datafiles, 'mean_chain_length', 'success_prob', make_success_prob, semilogy=True, base='test_gnp_hardness_.425')


    # 'test_gnp_hardness_.3'
    # first experiment label: 'test_gnp_hardness_.2'
    # first experiment wrapper ID: 5965d331c80ee230fe4b901f

    # create_df(ObjectId("595390c6c80ee26657faebc0"), db)#.to_pickle('multiembed.pklz')
    # create_df(ObjectId("59568cccc80ee2405ed733fd"), db).to_pickle('harness_test.pklz')
    # create_df(ObjectId("595aa412c80ee263579b25e1"), db).to_pickle('full_harness_test.pklz')
    # print munge('full_hardness_test.pklz', db)
    # munge('multiembed.pklz', db)
    # make_success_prob('munged_full_hardness_test.pklz', 'problem.params.p')
    analyze('munged_multiembed.pklz', 'mean_chain_length', 'closest_to_record')