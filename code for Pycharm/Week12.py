##############################################################
"""                       Week 12                          """
##############################################################


import numpy as np

import sympy as sp

""" 난수 """
np.random.normal(loc = 0.0, scale = 1.0, size = None)
np.random.uniform(low = 0.0, high = 1.0, size = None)


""" Multivariate """

# Rosenbrock
def f(x, a = 1, b = 5):
    y = (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2    
    return y

x = sp.IndexedBase('x')
gradients = np.array([sp.diff(f(x), x[i]) for i in range(2)])
grads = sp.lambdify(x, gradients, 'numpy')

x_ = [-2., 2.]

# Noisy descent
def NoisyDescent(f, grads, x, alpha, maxiter = 1000, TOL = 1E-4, verbose = True):
    
    y_prev = f(x)
    
    flag = True
    i = 1
    while flag:

        g = np.asarray(grads(x))
        g = - g / np.sqrt(np.dot(g, g))
        
        noise = np.random.normal(loc = 0, scale = 1. / i , size = g.shape)

        x = x + alpha * g + noise

        y = f(x)

        if verbose:
            print('{}: y {:.4f}, x {}'.format(i, y, x))
        
        if abs(y_prev - y) < TOL * (abs(y_prev) + TOL) or i >= maxiter:
            
            flag = False
        
        y_prev = y
        
        i += 1
        
    return i, y, x


NoisyDescent(f, grads, x_, alpha = 0.01)

y_best = 1E+8
x_best = [0., 0.]
    
for i in range(10):
    i, y, x = NoisyDescent(f, grads, x_, alpha = 0.01, verbose = False)
    if y < y_best:
        y_best, x_best = y, x

print(y_best, x_best)


# Simulated annealing
def SimulatedAnnealing(f, x, t, schedule, decay = 0.9, maxiter = 1000, TOL = 1E-4):

    y = f(x)
    y_best, x_best = y, x
    t0 = t

    flag = True
    i = 1
    while flag:    

        x_ = x + np.random.normal(size = len(x))
        y_ = f(x_)
        dy = y_ - y

        # clip
        t = max(t, 0.1)

        if dy < 0:
            dy = max(-10, dy)
        else:
            dy = min(10, dy)

        if y_ <= 0 or np.random.uniform() < min(np.exp(-dy / t), 1):
            x, y = x_, y_
            
        if y_ < y_best:
            y_best, x_best = y_, x_

        if schedule == 'logarithmic':
            t = t * np.log(2) / np.log(i + 1)
        elif schedule == 'exponential':
            t = decay * t
        elif schedule == 'fast':
            t = t0 / i            

        if i >= maxiter:
            
            flag = False
        
        i += 1
        
    return i, y_best, x_best



t = 1

SimulatedAnnealing(f, x_, t, 'logarithmic')
SimulatedAnnealing(f, x_, t, 'exponential')
SimulatedAnnealing(f, x_, t, 'fast')

















