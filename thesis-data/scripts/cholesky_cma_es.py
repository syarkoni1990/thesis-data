# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 19:02:37 2013

@author: wangronin
"""

import pdb
import numpy as np
from numpy.linalg import norm, cond
from numpy.random import randn, rand
from numpy import sqrt, eye, exp, dot, outer, inf, disp

def boundary_handling(x, lb, ub):
    """
    This function transforms x to t w.r.t. the low and high
    boundaries lb and ub. It implements the function T^{r}_{[a,b]} as
    described in Rui Li's PhD thesis "Mixed-Integer Evolution Strategies
    for Parameter Optimization and Their Applications to Medical Image 
    Analysis" as alorithm 6.
    
    """
    lb_index = np.isfinite(lb.flatten())
    up_index = np.isfinite(ub.flatten())
    
    valid = np.bitwise_and(lb_index,  up_index)
    
    LB = lb[valid, :]
    UB = ub[valid, :]

    y = (x[valid, :] - LB) / (UB - LB)
    I = np.mod(np.floor(y), 2) == 0
    yprime = np.zeros(np.shape(y))
    yprime[I] = np.abs(y[I] - np.floor(y[I]))
    yprime[~I] = 1.0 - np.abs(y[~I] - np.floor(y[~I]))

    x[valid, :] = LB + (UB - LB) * yprime
        
    return x
    
def cholesky_cma_es(dim, eval_budget, fitnessfunc, lb, ub, num_samples, **params):
    """ The (1+1)-Cholesky-CMA-ES as described in "A Computational 
    Efficient Covariance Matrix Update and a (1+1)-CMA for Evolution 
    Strategies" by Igel, Suttorp and Hansen
    
    """
    # Grabbing possible params:
    with_restarts = params.get('with_restarts', False) # reset sigma when too small; either True or False
    initial_offsets = params.get('initial_offsets', None) # initial offset values; either None or 'uniform'

    # print 'with restarts? ', with_restarts
    # print 'initial offsets? ', initial_offsets
    # exit()

    # Strategy parameters
    A = eye(dim)           # Cholesky-decomposition of covariance matrix
    ps = 2. / 11
    cp = 1. / 12
    ca = sqrt(1.0 - 2. / (dim ** 2. + 6))
    p_target = 2. / 11
    p_thresh = 11. / 25 
    damps = 1 + 1.0 / dim 
#    chiN = dim**.5 * (1-1./(4*dim)+1./(21*dim**2))
#    scale = chiN / 3.0

    history = dict(
        fitness=list(),
        opt_fitness=list(),
        # param=list(),
        opt_param=list(),
        A=list(),
        sigma=list(),
    )

    # Initialize dynamic (internal) strategy parameters and constants
    # reshape the arrays
    ub, lb = np.array(ub).reshape(-1, 1), np.array(lb).reshape(-1, 1)
    # set initial sigma
    sigma = np.max(ub - lb) / 10.

    # set initial offset positions
    if initial_offsets == 'uniform':
        xopt = parent = (ub - lb) * rand(dim, 1) + lb
    else:
        xopt = parent = np.zeros((dim,1))

    pfitness = fopt = fitnessfunc(xopt) 

    # -------------------- Generation Loop --------------------------------
    evalcount = num_samples
    while evalcount + num_samples < eval_budget:
        
        # Mutation
        z = randn(dim, 1) 
        offspring = parent + sigma * dot(A, z)
        offspring = boundary_handling(offspring, lb, ub)
        
        # Evaluation
        fitness = fitnessfunc(offspring)

        # print fitness

        evalcount += num_samples
        is_success = int(fitness < pfitness)
        
        # Cumulation of the success rate
        ps = (1.0 - cp) * ps + cp * is_success 
        
        # Adapt step size sigma, or restart sigma
        sigma *= exp((ps - (p_target / (1.0 - p_target)) * (1.0 - ps)) / damps)
        if sigma<=0.01 and with_restarts:
            sigma = np.max(ub - lb) / 10.
        
        # Update the parent
        # Adapt the covariance matrix
        if is_success:
            parent = offspring 
            pfitness = fitness 
            
            if ps < p_thresh:
                l = norm(z) ** 2.0
                A = ca * A + (ca / l) * (sqrt(1.0 + ((1. - ca ** 2.) * l) / (ca ** 2.)) - 1.) \
                    * dot(dot(A, z), z.T)
        
        # Check for degenerated parameters
        degenerated = (sigma > 1e16) or (sigma < 1e-16) or \
            (cond(A) > 1e16) or (cond(A) < 1e-16) or np.any(A == inf) 
        
        if degenerated:
            disp('parameters degenerated => reinitialize parameters')  
            sigma = np.max(ub - lb) / 10.
            A = eye(dim) 
            ps = 2.0 / 11
        
        if fitness < fopt:
            xopt = offspring 
            fopt = fitness


        history['A'].append(A.tolist())
        history['fitness'].append(fitness)
        history['opt_fitness'].append(fopt)
        # history['param'].append(offspring.tolist())
        history['opt_param'].append(xopt.tolist())
        history['sigma'].append(sigma.tolist())

    # return xopt, fopt, evalcount, history
    history['A'] = history['A'][-1]
    return history

    
if __name__ == '__main__':
    
    def sphere(x):
        return np.sum(x ** 2.)
    
    def ellipsoid(x):
        N = len(x)
        c = np.array([3. ** (i / (N - 1.)) for i in np.arange(0, N)]).reshape(-1, 1)
        f = np.sum((c * x) ** 2.)
        return f
    
    def schwefel(x):
        cum = np.cumsum(x)
        p = cum ** 2.
        return np.sum(p)
    
    dim = 30
    xopt, fopt, evalcount = cholesky_cma_es(dim, 1000, ellipsoid, [-5] * dim, [5] * dim)

    print xopt
    print fopt
    print evalcount
