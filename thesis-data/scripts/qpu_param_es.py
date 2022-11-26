__author__ = 'syarkoni'

# -*- coding: utf-8 -*-

"""

Created on Fri Oct 12 22:20:06 2012



@author: Hao Wang

"""



import pdb

import numpy as np

from numpy.random import rand, randn

# import matplotlib as mpl
# mpl.use('TkAgg')

import matplotlib.pyplot as plt

def dr1_es(dim, eval_budget, fitnessfunc, lb, ub, num_samples):

    """

    Derandomized mutative step-size control evolution strategy algorithm

    For minimization problem...



    Parameters

    ----------

        dim : int,

            dimension of the problem space

        eval_budget : int,

            number of the evaluations offered

        fitnessfunc : callable,

            function handle of the object function (probably mean or min energy, depending on the application)

        lb : list or 1-d array

            lower bounds

        ub : list or 1-d array

            upper bounds

    Returns

    -------

        xopt : the best individual ever found

        fopt : the fitness value of the best individual ever found

        evalcount : the number of function evaluation costed when the algorithm terminates

    """


    #TODO: run dr1 for long time and check that algorithm actually converges for small problem sizs
    #TODO: run for same num generations at larger problem sizes to determine if convergence rate is just slower
    #TODO: decrease step sizes, try 1/15 instead of 1/6.
    #TODO: do a "calibration" run to learn the anneal offsets, then another run to find minimum
    #TODO: possibly add gaussian process method to compare against
    #TODO: add a db collection "hyper_optimizer" or something to store results with hyper parameter optimization
    #TODO: add random restarts to the ESs



    # Strategy parameters

    _lambda = 10
    # _lambda is the number of "independent trials" at the current param configuration.

    beta_scal = 1.0 / dim

    beta = np.sqrt(beta_scal)

    alpha = 5 / 7.

    # initial individual step-sizes


    ub, lb = 0.99*np.array(ub), 0.99*np.array(lb)

    sigma = np.array([(ub - lb) / 6. for _ in range(_lambda)])

    # xopt = x = ub * np.ones(dim)
    xopt = x = (ub - lb) * rand(dim) + lb

    fopt = fitnessfunc(xopt)

    # declare history dict for plotting
    history = dict(
        fitness_eval=list(),
        param_hist=list(),
        opt_param_hist=list(),
        opt_fitness_hist=list(),
        sigma=list(),
    )

    # Evolution loop

    evalcount = num_samples

    while evalcount + _lambda < eval_budget:
        print 'doing the loop'
        # Mutation

        ksi = alpha * np.ones((_lambda, 1))

        pos = rand(_lambda) > .5

        ksi[pos] = 1. / alpha

        z = randn(_lambda, dim)

        offsprings = x + ksi * sigma * z
        history['param_hist'].append(offsprings.tolist())
        # This loop clips the bounds if they're violated, which they shouldn't be anymore
        for i, sub in enumerate(offsprings):
            for j, val in enumerate(sub):
                if val > ub[j]:
                    offsprings[i][j] = 0.99*ub[j]
                if val < lb[j]:
                    offsprings[i][j] = 0.99*lb[j]

        # Evaluation
        fitness = [fitnessfunc(_) for _ in offsprings]
        print fitness

        history['fitness_eval'].append(fitness)
        history['sigma'].append(sigma.tolist())

        evalcount += _lambda * num_samples

        # Comma selection
        try:
            sel = np.flatnonzero(fitness == min(fitness))[0]
        except ValueError as e:
            print fitness
            exit()
        x = offsprings[sel, :]

        # Comma selection- original

        # sel, = np.nonzero(fitness == min(fitness))[0]
        # x = offsprings[sel, :]




        # Step-size adaptation

        z_ = np.exp(np.abs(z[sel, :]) - np.sqrt(2 / np.pi))

        sigma = ksi[sel] ** beta * z_ ** beta_scal * sigma

        if fopt > fitness[sel]:

            fopt = fitness[sel]

            xopt = offsprings[sel, :]

        history['opt_param_hist'].append(xopt.tolist())
        history['opt_fitness_hist'].append(fopt)

    return xopt, fopt, evalcount, history



if __name__ == '__main__':
    pass
