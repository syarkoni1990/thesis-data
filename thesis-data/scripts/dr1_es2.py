# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 22:20:06 2012

@author: Hao Wang
"""

import pdb
import numpy as np
from numpy.random import rand, randn

def boundary_handling(x, lb, ub):
    """
    
    This function transforms x to t w.r.t. the low and high
    boundaries lb and ub. It implements the function T^{r}_{[a,b]} as
    described in Rui Li's PhD thesis "Mixed-Integer Evolution Strategies
    for Parameter Optimization and Their Applications to Medical Image 
    Analysis" as alorithm 6.
    
    """
    lb, ub = lb.flatten(), ub.flatten()
    
    lb_index = np.isfinite(lb)
    up_index = np.isfinite(ub)
    
    valid = np.bitwise_and(lb_index,  up_index)
    
    LB = lb[valid]
    UB = ub[valid]

    y = (x[:, valid] - LB) / (UB - LB)
    I = np.mod(np.floor(y), 2) == 0
    yprime = np.zeros(np.shape(y))
    yprime[I] = np.abs(y[I] - np.floor(y[I]))
    yprime[~I] = 1.0 - np.abs(y[~I] - np.floor(y[~I]))

    x[:, valid] = LB + (UB - LB) * yprime
        
    return x


def dr1_es(dim, eval_budget, fitnessfunc, lb, ub):
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
            function handle of the object function
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
    # Strategy parameters
    _lambda = 10
    beta_scal = 1.0 / dim
    beta = np.sqrt(beta_scal)
    alpha = 5 / 7.
    
    # initial individual step-sizes
    sigma = np.ones(dim) * 20
    ub, lb = np.array(ub), np.array(lb)
    xopt = x = (ub - lb) * rand(dim) + lb
    fopt = fitnessfunc(xopt)
    
    # Evolution loop
    evalcount = 1
    while evalcount + _lambda < eval_budget:
        # Mutation
        ksi = alpha * np.ones((_lambda, 1))
        pos = rand(_lambda) > .5
        ksi[pos] = 1. / alpha

        z = randn(_lambda, dim)
        offsprings = x + ksi * sigma * z
        
        # boundary handling: solution repair
        offsprings = boundary_handling(offsprings, lb, ub)
        
        # Evaluation
        fitness = [fitnessfunc(_) for _ in offsprings]
        evalcount += _lambda

        # Comma selection   
        sel, = np.nonzero(fitness == min(fitness))[0]
        x = offsprings[sel, :]
            
        # Step-size adaptation
        z_ = np.exp(np.abs(z[sel, :]) - np.sqrt(2 / np.pi))
        sigma = ksi[sel] ** beta * z_ ** beta_scal * sigma
        
        if fopt > fitness[sel]:
            fopt = fitness[sel]
            xopt = offsprings[sel, :]
            
    return xopt, fopt, evalcount
        
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
    
    dim = 5
    xopt, fopt, evalcount = dr1_es(dim, 2e3, ellipsoid, [-5] * dim, [5] * dim)

    print xopt
    print fopt
    print evalcount

    
    
