# Gradient Descent

import numpy as np

# Algorithm 4.3: p.62
def strong_backtracking(f, grads, x, d, alpha = 1.0, beta = 1E-4, sigma = 1E-1, verbose = False):
    
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
        
    if verbose:
        print('backtracking: alpha %.4f, y %.4f' % (alpha, y))
    
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

            
def GradientDescent(f, grads, x, alpha, maxiter = 1000, TOL = 1E-4, verbose = False):
    
    y_prev = f(x)
    
    flag = True
    i = 1
    
    while flag:
        
        g = np.asarray(grads(x))
        g = g / np.sqrt(np.dot(g, g))
        
        x = x + alpha * (-1 * g)
        
        y = f(x)

        if verbose:
            print('{}: y {:.4f}, x {}'.format(i, y, x))
        
        if abs(y_prev - y) < TOL * (abs(y_prev) + TOL) or i >= maxiter:
            
            flag = False
        
        y_prev = y
        
        i += 1
        
    return i, y, x
        
    

def ConjugateGradient(f, grads, x, method = 'FL', maxiter = 1000, TOL = 1E-4, verbose = False):
    
    y_prev = f(x)
    g_prev = np.asarray(grads(x))
    d = -1 * g_prev
    
    alpha =  strong_backtracking(f, grads, x, d, 1)
    x = x + alpha * d
    
    i = 1
    flag = True
    while flag:
        
        g = np.asarray(grads(x))
        
        if method == 'FL':
            beta = (np.dot(g, g)) / (np.dot(g_prev, g_prev)) # Fletcher_Reeves
        elif method == 'PR':
            beta = (np.dot(g, (g - g_prev))) / (np.dot(g_prev, g_prev)) # Polak-Ribiere
            beta = np.max(beta, 0)
        else:
            beta = (np.dot(g, g)) / (np.dot(g_prev, g_prev)) # Fletcher_Reeves
            
        d = - g + beta * d
        
        alpha =  strong_backtracking(f, grads, x, d, 1)

        x = x + alpha * d
        y = f(x)

        if verbose:
             print('{}: y {:.4f}, x {}'.format(i, y, x))
        
        if np.abs(y - y_prev) < TOL * (np.abs(y_prev) + TOL) or i >= maxiter:
            
            flag = False
        
        y_prev = y
        g_prev = g
        
        i += 1
        
    return i, y, x

