##############################################################
"""                       Week 11                          """
##############################################################


import numpy as np
import matplotlib.pyplot as plt

import sympy as sp


""" Univariate """

def f(x):
    y = x ** 5 - 8 * x **3 + 10 * x + 6
    return y

x = sp.Symbol('x')

df = sp.diff(f(x), x)
derv = sp.lambdify(x, df, 'numpy') # x, df: symbolic variable

df2 = sp.diff(df, x)
derv2 = sp.lambdify(x, df2, 'numpy') 

# Newton's method: univariate
def uni_newton(f, derv, derv2, x, maxiter = 1000, TOL = 1E-4, verbose = True):

    y_prev = f(x)
  
    flag = True
    i = 1
    while flag:
    
        x = x - derv(x) / derv2(x)

        y = f(x) 
        
        if verbose:
             print('{}: y {:.4f}, x {}'.format(i, y, x))
        
        if abs(y_prev - y) < TOL * (abs(y_prev) + TOL) or i >= maxiter:
            
            flag = False
        
        y_prev = y
        
        i += 1
        
    return i, y, x
        

x_ = -1.
uni_newton(f, derv, derv2, x_)    


# Secant method: univariate
def secant(f, derv, x0, x1, epsilon = 1E-7, maxiter = 1000, TOL = 1E-4, verbose = True):

    y_prev = f(x0)
  
    flag = True
    i = 1
    while flag:
        
        x = x1 - ((x1 - x0) / (derv(x1) - derv(x0) + epsilon)) * derv(x1)

        y = f(x) 
        
        if verbose:
             print('{}: y {:.4f}, x {}'.format(i, y, x1))
        
        if abs(y_prev - y) < TOL * (abs(y_prev) + TOL) or i >= maxiter:
            
            flag = False
        
        y_prev, x0, x1 = y, x1, x
        
        i += 1
        
    return i, y, x1

secant(f, derv, -1., -0.9)


""" Multivariate """

# Example 6.1: p.90
# Booth function : global minimum 0 at [1., 3.]
def draw_booth(levels):

    boothfunction = lambda x1, x2: (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2
    
    X1, X2 = np.meshgrid(np.linspace(-10.0, 10.0, 400), np.linspace(-10.0, 10.0, 400))
    
    Y = boothfunction(X1, X2)
    
    plt.figure(figsize = (8, 8))
    
    plt.contour(X1, X2, Y, np.logspace(-2.0, 2.0, levels, base = 10)) #, cmap = 'gray')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

draw_booth(10)


def f(x):
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


x = sp.IndexedBase('x')
gradients = np.array([sp.diff(f(x), x[i]) for i in range(2)])
grads = sp.lambdify(x, gradients, 'numpy')

hessian = np.asarray([[sp.diff(gradients[i], x[j]) for i in range(2)] for j in range(2)])
hess = sp.lambdify(x, hessian, 'numpy')

x_ = [1., 1.]
x_ - np.dot(np.linalg.inv(hess(x_)), grads(x_))


def newton(f, grads, hessian, x, maxiter = 1000, TOL = 1E-4, verbose = True):

    y_prev = f(x)
  
    flag = True
    i = 1
    while flag:
    
        x = x - np.dot(np.linalg.inv(hess(x)), grads(x))

        y = f(x) 
        
        if verbose:
             print('{}: y {:.4f}, x {}'.format(i, y, x))
        
        if abs(y_prev - y) < TOL * (abs(y_prev) + TOL) or i >= maxiter:
            
            flag = False
        
        y_prev = y
        
        i += 1
        
    return i, y, x


newton(f, grads, hess, x_)


# Rosenbrock
def f(x, a = 1, b = 5):
    y = (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2    
    return y


x = sp.IndexedBase('x')
gradients = np.array([sp.diff(f(x), x[i]) for i in range(2)])
grads = sp.lambdify(x, gradients, 'numpy')

hessian = np.asarray([[sp.diff(gradients[i], x[j]) for i in range(2)] for j in range(2)])
hess = sp.lambdify(x, hessian, 'numpy')

x_ = [-2., 2.]
newton(f, grads, hess, x_)











  