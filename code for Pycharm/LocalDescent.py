# Local Descent

import numpy as np

import Bracketing


# Algorithm 4.1: p.54
def line_search(f, x, d):
    
    def obj(alpha):
        
            return f(x + alpha * d)
            
    
    a, b = Bracketing.bracket_minimum(obj, 0.)
    
    _, alpha, _ = Bracketing.golden_section_search(obj, 0.)
    
    return alpha, x + alpha * d


# Algorithm 4.2: p.56
def backtracking_line_search(f, grads, x, d, alpha = 10, p = 0.5, beta = 1E-4, verbose = False):
    
    y, g = f(x), grads(x)
    
    i = 1
    while f(x + alpha * d) > y + beta * alpha * np.dot(g, d): # g.T @ d
        
        alpha *= p

        if verbose:
            print('%d: alpha = %.4f' % (i, alpha))
        
        i += 1
        
    return alpha


# Algorithm 4.3: p.62
def strong_backtracking(f, gradient, x, d, alpha = 1.0, beta = 1E-4, sigma = 1E-1, verbose = False):
    
    y0, g0, y_prev, alpha_prev = f(x), np.dot(gradient(x), d), np.nan, 0.0
    alpha_lo, alpha_hi = np.nan, np.nan

    # bracket phase
    while True:
        
        y = f(x + alpha * d)
        
        if y > y0 + beta * alpha * g0 or (not(np.isnan(y_prev)) and y >= y_prev):
            alpha_lo, alpha_hi = alpha_prev, alpha
            break
        
        g = np.dot(gradient(x + alpha * d), d)
        
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
            g = np.dot(gradient(x + alpha * d), d)
            
            if abs(g) <= -sigma * g0:
                return alpha
            elif g * (alpha_hi - alpha_lo) >= 0.0:
                alpha_hi = alpha_lo
            
            alpha_lo = alpha


# local descent with backtracking line search
def local_descent_backtracking(f, grads, x_, alpha = 1, TOL = 1E-8, verbose = False):

    d_ = -1 * np.array(grads(x_))
    
    alpha = backtracking_line_search(f, grads, x_, d_, alpha)
    
    if verbose:
        print(alpha, x_, f(x_))
    
    y_prev = f(x_)
    
    i = 1
    flag = True
    while flag:
    
        x_ = x_ + alpha * d_
        d_ = -1 * np.array(grads(x_))
    
        alpha = backtracking_line_search(f, grads, x_, d_, alpha)
    
        y_ = f(x_)
        
        diff = np.abs(y_ - y_prev)
        
        if verbose:
            print(i, alpha, x_, f(x_), diff)

        if diff < TOL * (abs(y_prev) + TOL):
            flag = False
   
        y_prev = y_
        
        i += 1

    return i, x_, y_


# local descent with strong backtracking
def local_descent_strong_backtracking(f, grads, x_, alpha = 1, TOL = 1E-8, verbose = False):

    d_ = -1 * np.array(grads(x_))
    
    alpha = strong_backtracking(f, grads, x_, d_, alpha)
    
    if verbose:
        print(alpha, x_, f(x_))
    
    y_prev = f(x_)
    
    i = 1
    flag = True
    while flag:
    
        x_ = x_ + alpha * d_
        d_ = -1 * np.array(grads(x_))
    
        alpha = strong_backtracking(f, grads, x_, d_, alpha)
    
        y_ = f(x_)
        
        diff = np.abs(y_ - y_prev)
        
        if verbose:
            print(i, alpha, x_, f(x_), diff)

        if diff < TOL * (abs(y_prev) + TOL):
            flag = False
    
        y_prev = y_
        i += 1
    
    return i, x_
