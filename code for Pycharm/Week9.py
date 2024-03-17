##############################################################
"""                        Week 9                          """
##############################################################


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 문제해결과제(1)
def f(x):

    y = x ** 5 - 8 * x ** 3 + 10 * x + 6

    return y


# 목적함수 탐색
x = np.arange(-3.0, 3.0, 0.1)

plt.figure(figsize = (10, 5))
fig = sns.lineplot(x = x, y = [f(xi) for xi in x])
fig.set(xlabel = 'x', ylabel = 'f(x)', title = 'y = x ** 5 - 8 * x ** 3 + 10 * x + 6')
plt.show()


import Bracketing


# find bracket
Bracketing.bracket_minimum(f, 0)
Bracketing.bracket_minimum(f, -1)
Bracketing.bracket_minimum(f, -2)
Bracketing.bracket_minimum(f, -3)
Bracketing.bracket_minimum(f, 1)
Bracketing.bracket_minimum(f, 2)
Bracketing.bracket_minimum(f, 3)
Bracketing.bracket_minimum(f, 4)


# trifold
Bracketing.trifold_search(f, 1, verbose = True)
Bracketing.trifold_search(f, -1, verbose = True)


# golden section
Bracketing.golden_section_search(f, 1)
Bracketing.golden_section_search(f, -1)


import sympy as sp

# Example 3.1 : univariate

x = sp.Symbol('x')
df = sp.diff(f(x), x)
print(df)

derv = sp.lambdify(x, df, 'numpy') # x, df: symbolic variable

x = np.arange(-3.0, 3.0, 0.1)

plt.figure(figsize = (10, 5))
fig = sns.lineplot(x = x, y = [derv(xi) for xi in x])
fig.set(xlabel = 'x', ylabel = 'f(x)', title = "y\' = 5 * x**4 - 24 * x**2 + 10")
fig.axhline(y = 0, color = 'r')
plt.show()


Bracketing.bracket_sign_change(derv, 1 - 1E-6, 1 + 1E-6)

Bracketing.bisection(derv, 1, verbose = True)


# Local Descent
import LocalDescent as ld

x = 0
d = -1
ld.line_search(f, x, d)
    
   
def f(x):
    y = x[0] ** 5 - 8 * x[0] **3 + 10 * x[0] + 6
    return y


x = sp.IndexedBase('x')
gradients = np.array([sp.diff(f(x), x[i]) for i in range(1)])
grads = sp.lambdify(x, gradients, 'numpy')

x_ = np.array([-1])
ld.local_descent_backtracking(f, grads, x_, 1)

x_ = np.array([-2])
ld.local_descent_backtracking(f, grads, x_, 1)


x_ = np.array([-1])
ld.local_descent_strong_backtracking(f, grads, x_, 1)

x_ = np.array([-2])
ld.local_descent_strong_backtracking(f, grads, x_, 1)


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


draw_rosenbrock(1, 5, 10)
draw_rosenbrock(1, 100, 5)



import GradientDescent as gd

# Rosenbrock
def f(x, a = 1, b = 5):
    y = (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2    
    return y

x = sp.IndexedBase('x')
gradients = np.array([sp.diff(f(x), x[i]) for i in range(2)])
grads = sp.lambdify(x, gradients, 'numpy')


x_ = np.array([-2, 2])
alpha = 1E-2

gd.GradientDescent(f, grads, x_, alpha)

gd.ConjugateGradient(f, grads, x_)


############################################
""" 문제해결형 과제 (2) """
############################################

""" Branin Function """

def draw_branin(levels):

    a = 1
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    
    braninfunction = lambda x1, x2: a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    
    X1, X2 = np.meshgrid(np.linspace(-5., 20., 100), np.linspace(-5.0, 20.0, 100))
    
    Y = braninfunction(X1, X2)
    
    plt.figure(figsize = (8, 8))
    
    plt.contour(X1, X2, Y, np.logspace(-5.0, 20.0, levels, base = 10)) #, cmap = 'gray')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


draw_branin(100)


# Branin Function
def f(x, a = 1, b = 5.1 / (4 * np.pi ** 2), c = 5 / np.pi, r = 6, s = 10, t = 1 / (8 * np.pi)):
    
    y = a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * sp.cos(x[0]) + s

    return y


x = sp.IndexedBase('x')
gradients = np.array([sp.diff(f(x), x[i]) for i in range(2)])
grads = sp.lambdify(x, gradients, 'numpy')

x_ = np.array([5., 5.])
gd.GradientDescent(f, grads, x_, alpha)

gd.ConjugateGradient(f, grads, x_, verbose = True)



""" Momentum """
def momentum(f, grads, x, alpha, beta = 0.9, maxiter = 1000, TOL = 1E-4, verbose = False):
 
    v = -alpha * np.asarray(grads(x))
    x = x + v
 
    y_prev = f(x)
 
    flag = True
    i = 1
    while flag:
        
        v = beta * v - alpha * np.asarray(grads(x))
        x = x + v
      
        y = f(x)

        if verbose:
             print('{}: y {:.4f}, x {}'.format(i, y, x))
 
        if abs(y_prev - y) < TOL * (abs(y_prev) + TOL) or i >= maxiter:
            
            flag = False
 
        y_prev = y
        
        i += 1
 
    return i, y, x
 
    
momentum(f, grads, x_, alpha = 1E-2, verbose = True)

""" Nesterov (1983) """
def nesterov(f, grads, x, alpha, beta = 0.9, maxiter = 1000, TOL = 1E-4, verbose = True):
 
    v = -alpha * np.asarray(grads(x))
    x = x + v

    y_prev = f(x)
    
    
    flag = True
    i = 1
    while flag:

        v = beta * v - alpha * np.asarray(grads(x + beta * v))
        x = x + v
           
        y = f(x)

        if verbose:
             print('{}: y {:.4f}, x {}'.format(i, y, x))
        
        if abs(y_prev - y) < TOL * (abs(y_prev) + TOL) or i >= maxiter:
            
            flag = False
        
        y_prev = y
        
        i += 1
        
    return i, y, x
        

nesterov(f, grads, x_, alpha = 1E-2, verbose = True)


""" Adagrad: Adaptive subgradient (2011) """
def adagrad(f, grads, x, alpha, epsilon = 1E-7, maxiter = 1000, TOL = 1E-4, verbose = True):

    s = np.zeros_like(x)

    y_prev = f(x)
 
    flag = True
    i = 1
    while flag:
 
        g = np.asarray(grads(x))
        s += (g * g)
        
        x = x - alpha * g / np.sqrt(s + epsilon)

        y = f(x) 
        
        if verbose:
             print('{}: y {:.4f}, x {}'.format(i, y, x))
        
        if abs(y_prev - y) < TOL * (abs(y_prev) + TOL) or i >= maxiter:
            
            flag = False
        
        y_prev = y
        
        i += 1
        
    return i, y, x
        
     
adagrad(f, grads, x_, alpha = 1E-1, verbose = True)














    














