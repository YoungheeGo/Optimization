##############################################################
"""                         Week 6                         """
##############################################################

import numpy as np

from IPython.display import Image
Image("./fig6-1.png", width = 640)#, height = 300)

Image("./fig6-2.png", width = 640)#, height = 300)

# Example 4.2: p. 58
def f(x):
    
    y = x[0] ** 2 + x[0] * x[1] + x[1] ** 2
    
    return y

def gradient(x):
    
    return [2 * x[0] + x[1], 2 * x[1] + x[0]]

x = np.asarray([1, 2])
d = np.asarray([-1.1, -1.2])

print(f(x))
print(gradient(x))

# Algorithm 4.3: p. 62
def strong_backtracking(f, gradient, x, d, alpha = 1, beta = 1E-4, sigma = 1E-1):
    
    y0, g0, y_prev, alpha_prev = f(x), np.dot(gradient(x), d), np.nan, 0.0
    alpha_lo, alpha_hi = np.nan, np.nan
    
    # bracket phase
    while True:
        
        y = f(x + alpha * d)
        
        if (y > y0 + beta * alpha * g0) or (not(np.isnan(y_prev)) and y >= y_prev):
            alpha_lo, alpha_hi = alpha_prev, alpha
            break
        
        g = np.dot(gradient(x + alpha * d), d)
        
        if np.abs(g) <= -sigma * g0:
            return alpha
        elif g >= 0:
            alpha_lo, alpha_hi, alpha, alpha_prev
            break
        
        y_prev, alpha_prev, alpha = y, alpha, 2 * alpha
        
    print('bracket: %.4f, %.4f' % (alpha_lo, alpha_hi))
          
    # zoom phase
    y_lo = f(x + alpha_lo * d)

    while True:

        alpha = 0.5 * (alpha_lo + alpha_hi)        
        y = f(x + alpha * d)
        
        if (y > y0 + beta * alpha * g0) or (y >= y_lo):
            alpha_hi = alpha
        else:
            g = np.dot(gradient(x + alpha * d), d)
        
            if abs(g) <= -sigma * g0:
                return alpha
            elif g * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
                
            alpha_lo = alpha
        
alpha = strong_backtracking(f, gradient, x, d)
        
print(alpha)
print(x + alpha * d)        
print(f(x + alpha * d))        
