##############################################################
""" Week 8 """
##############################################################

# Gradient Descent

import numpy as np

# Symbolic differenciation

import sympy as sp

# Rosenbrock
def f(x, a = 1, b = 5):
    y = (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2    
    return y

x = sp.IndexedBase('x')
gradients = np.array([sp.diff(f(x), x[i]) for i in range(2)])
grads = sp.lambdify(x, gradients, 'numpy')


def GradientDescent(f, grads, x, alpha, maxiter = 1000, TOL = 1E-4):
    
    y_prev = f(x)
    
    flag = True
    i = 1
    
    while flag:
        
        g = np.asarray(grads(x))
        g = g / np.sqrt(np.dot(g, g))
        d = -1 * g
        
        x = x + alpha * d
        
        y = f(x)
        
        print('{}: y {:.4f}, x {}'.format(i, y, x))
        
        if abs(y_prev - y) < TOL * (abs(y_prev) + TOL) or i > maxiter:
            
            flag = False
            
        y_prev = y
        
        i += 1
        
    return x

x_ = np.array([-2, 2])
alpha = 1E-2

GradientDescent(f, grads, x_, alpha)

            
 # Algorithm 4.3: p.62
def strong_backtracking(f, grads, x, d, alpha = 1.0, beta = 1E-4, sigma = 1E-1):
    
    y0, g0, y_prev, alpha_prev = f(x), np.dot(grads(x), d), np.nan, 0.0
    alpha_lo, alpha_hi = np.nan, np.nan

    # bracket phase
    while True:
        
        y = f(x + alpha * d)
        
        if y > y0 + beta * alpha * g0 or (not(np.isnan(y_prev)) and y >= y_prev):
            alpha_lo, alpha_hi = alpha_prev, alpha
            break
        
        g = np.dot(grads(x + alpha * d), d)
        
        if np.abs(g) <= -sigma * g0:
            return alpha
        elif g >= 0:
            alpha_lo, alpha_hi = alpha, alpha_prev
            break
        
        y_prev, alpha_prev, alpha = y, alpha, 2 * alpha
        
    #print('backtracking: alpha %.4f, y %.4f' % (alpha, y))
    
    # zoom phase
    y_lo = f(x + alpha_lo * d)
    
    while True:

        alpha = 0.5 * (alpha_lo + alpha_hi)
        y = f(x + alpha * d)
        
        if (y > y0 + beta * alpha * g0 ) or (y >= y_lo):
            alpha_hi = alpha
        else:
            g = np.dot(grads(x + alpha * d), d)
            
            #print(abs(g), -sigma * g0)
            
            if abs(g) <= -sigma * g0:
                return alpha
            elif g * (alpha_hi - alpha_lo) >= 0.0:
                alpha_hi = alpha_lo
            
            alpha_lo = alpha

                
def ConjugateGradient(f, grads, x, maxiter = 1000, TOL = 1E-4):

    y_prev = f(x)
    g_prev = np.asarray(grads(x))
    d = -1 * g_prev

    alpha = strong_backtracking(f, grads, x, d, 1)
    x = x + alpha * d
    
    flag = True
    i = 1
    while flag:
        
        g = np.asarray(grads(x))
        
        #beta = (np.dot(g, g)) // (np.dot(g_prev, g_prev)) # Fleecher
        beta = (np.dot(g, (g - g_prev))) / (np.dot(g_prev, g_prev)) # Polak-Ribiere
        beta = np.max(beta, 0)

        d = - g + beta * d
        
        alpha = strong_backtracking(f, grads, x, d, 1)
        
        x = x + alpha * d
        y = f(x)
        
        print('{}: y {:.4f}, x {}'.format(i, y, x))
          
        
        if np.abs(y - y_prev) < TOL * (np.abs(y_prev) + TOL) or i > maxiter:
            
            flag = False
    
        y_prev = y
        g_prev = g
 
        i += 1
        
    return x


ConjugateGradient(f, grads, x_)

""" 교과목 포트폴리오 """
   
""" keystroke 자료 좀 부탁해요 ㅠㅠㅠ """
  
        
        