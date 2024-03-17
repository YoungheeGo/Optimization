##############################################################
"""                       Week 10                          """
##############################################################


import numpy as np
import matplotlib.pyplot as plt

import sympy as sp

# Gradient Descent
def draw_rosenbrock(a, b, levels):

    rosenbrockfunction = lambda x1, x2: (a - x1) ** 2 + b * (x2 - x1 ** 2) ** 2
    
    X1, X2 = np.meshgrid(np.linspace(-2.0, 2.0, 400), np.linspace(-2.0, 2.0, 400))
    
    Y = rosenbrockfunction(X1, X2)
    
    plt.figure(figsize = (8, 8))
    
    plt.contour(X1, X2, Y, np.logspace(-2.0, 2.0, levels, base = 10)) #, cmap = 'gray')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


draw_rosenbrock(1, 100, 5)


import GradientDescent as gd


# Rosenbrock
def f(x, a = 1, b = 100):
    y = (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2    
    return y


x = sp.IndexedBase('x')
gradients = np.array([sp.diff(f(x), x[i]) for i in range(2)])
grads = sp.lambdify(x, gradients, 'numpy')


x_ = np.array([-2., 2.])


gd.GradientDescent(f, grads, x_, alpha = 1E-1, verbose = True)

gd.ConjugateGradient(f, grads, x_)

gd.momentum(f, grads, x_, alpha = 7E-4, verbose = True)

gd.nesterov(f, grads, x_, alpha = 7E-4, verbose = True)


""" Huge-variate : e.g. Deep Learning """
gd.adagrad(f, grads, x_, alpha = 3.05E-0, verbose = True)


""" RMSProp: Geoffrey Hinton """
def rmsprop(f, grads, x, alpha, decay = 0.9, epsilon = 1E-7, maxiter = 1000, TOL = 1E-4, verbose = True):

    s = np.zeros_like(x)

    y_prev = f(x)
  
    flag = True
    i = 1
    while flag:
        
        g = np.asarray(grads(x))
        s = decay * s + (1. - decay) * (g * g)

        RMS = np.sqrt(s + epsilon)
        x = x - alpha * g / RMS
        
        y = f(x) 
        
        if verbose:
             print('{}: y {:.4f}, x {}'.format(i, y, x))
        
        if abs(y_prev - y) < TOL * (abs(y_prev) + TOL) or i >= maxiter:
            
            flag = False
        
        y_prev = y
        
        i += 1
        
    return i, y, x
                
        
rmsprop(f, grads, x_, alpha = 1.01E-0, verbose = True)
        
        

""" Adadelta (2012) """
def adadelta(f, grads, x, rho = 0.95, epsilon = 1E-7, maxiter = 1000, TOL = 1E-4, verbose = True):

    s = np.zeros_like(x)
    u = np.zeros_like(x)

    y_prev = f(x)
  
    flag = True
    i = 1
    while flag:
        
        g = np.asarray(grads(x))
        s = rho * s + (1. - rho) * (g * g)
         
        delta = (np.sqrt(u + epsilon) / np.sqrt(s + epsilon)) * g
         
        u = rho * u + (1. - rho) * delta * delta
        
        x = x - delta 
        
        y = f(x) 
        
        if verbose:
             print('{}: y {:.4f}, x {}'.format(i, y, x))
        
        if abs(y_prev - y) < TOL * (abs(y_prev) + TOL) or i >= maxiter:
            
            flag = False
        
        y_prev = y
        
        i += 1
        
    return i, y, x
        
        
adadelta(f, grads, x_, rho = 0.9, verbose = True)
adadelta(f, grads, x_, rho = 0.5, verbose = True)
adadelta(f, grads, x_, rho = 0.03, maxiter = 10000, verbose = True)


""" Adam: Adaptive moment estimation (2015) """
def adam(f, grads, x, alpha, rho1 = 0.9, rho2 = 0.99, epsilon = 1E-7, maxiter = 1000, TOL = 1E-4, verbose = True):

    k = 0
    v = np.zeros_like(x)
    s = np.zeros_like(x)
 
    y_prev = f(x)
 
    flag = True
    i = 1
    while flag:
        
        g = np.asarray(grads(x))
     
        v = rho1 * v + (1. - rho1) * g
        s = rho2 * s + (1. - rho2) * (g * g)

        k += 1
     
        v_hat = v / (1 - rho1 ** k)
        s_hat = s / (1 - rho2 ** k)
 
        x = x - alpha * (v_hat / np.sqrt(s_hat + epsilon)) 
    
        y = f(x) 
        
        if verbose:
             print('{}: y {:.4f}, x {}'.format(i, y, x))
        
        if abs(y_prev - y) < TOL * (abs(y_prev) + TOL) or i >= maxiter:
            
            flag = False
        
        y_prev = y
        
        i += 1
        
    return i, y, x 
    
      
adam(f, grads, x_, alpha = 3E-0, verbose = True)
 









