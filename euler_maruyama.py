#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

def euler_maruyama(u0,T,N,f,g):
    """
    u0 - initial condition
    T - end time
    N - number of time intervals
    f - function handle
    g - function handle
    """
    dim = len(u0)
    solutions = []
    solutions.append(u0)
    times, dt = np.linspace(0,T,N,retstep = True)
    for t in times:
        u0 += f(u0)*dt + g(u0) * np.random.normal(0,np.sqrt(dt),dim)
        solutions.append(u0)
    return np.vstack(solutions)

solutions = euler_maruyama(u0,T,N,f,g)
plt.hist(solutions)
plt.show()





