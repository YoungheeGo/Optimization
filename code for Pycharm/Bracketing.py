# Bracketing Module

import numpy as np


def bracket_minimum(f, x, s = 1E-2, k = 2.0, verbose = False):
    
    a, ya = x, f(x)
    b, yb = a + s, f(a + s)

    if verbose:
        print('init: (a:%.4f, b:%.4f) (ya:%.4f, yb:%.4f)' % (a, b, ya, yb))
    
    
    if yb > ya:
        a, b = b, a
        ya, yb = yb, ya
        s = -s
        
    while True:
        c, yc = b + s, f(b + s)

        if verbose:
            print('step: (a:%.4f, b:%.4f, c:%.4f) (ya:%.4f, yb:%.4f, yc:%.4f)' % (a, b, c, ya, yb, yc))

        if yc > yb:

            return (a, c) if a < c else (c, a)

        else:
            a, ya, b, yb = b, yb, c, yc
            s *= k
    

# braketing by three fold
def trifold_search(f, x, verbose = False):
    
    a, b = bracket_minimum(f, x)
    
    if verbose:
        print('init:(a:%.4f, b:%.4f)' % (a, b))
    
    distance = abs(a - b)
    
    i = 1
    while distance > 1E-6:
    
        x1 = a + (1.0 / 3.0) * distance
        x2 = a + (2.0 / 3.0) * distance
        
        y1, y2 = f(x1), f(x2)
        
        if y1 > y2:
            a, b = x1, b
        else:
            a, b = a, x2
            
    
        distance = abs(a - b)

        if verbose:
            print('%d:(a:%.4f, b:%.4f)' % (i, a, b))
        
        i += 1
    
    x = a + 0.5 * abs(a - b)
    y = f(x)
    
    return i, x, y


# Algorithm 3.2: p. 39
def fibonacci_search(f, x, n, epsilon = 1E-2, verbose = False):
    
    a, b = bracket_minimum(f, x)
    
    if verbose:
        print('init:(a:%.4f, b:%.4f)' % (a, b))
    
    psi = 0.5 * (1. + np.sqrt(5))
    s = (1. - np.sqrt(5)) / (1. + np.sqrt(5))
    
    rho = 1. / psi * ((1. - s ** (n + 1)) / (1. - s ** n))
    d = rho * b + (1. - rho) * a
    
    yd = f(d)
    
    for i in range(1, n):
        if i == n - 1:
            c = epsilon * a + (1. - epsilon) * d
            
        else:
            c = rho * a + (1. - rho) * b
        yc = f(c)
        if yc < yd:
            b, d, yd = d, c, yc
        else:
            a, b = b, c
    
        rho = 1. / psi * ((1. - s ** (n - i + 1)) / (1. - s ** (n - i)))

        pa, pb = (a, b) if a < b else (b, a)
        
        if verbose:
            print('%d:(a:%.4f, b:%.4f)' % (i, pa, pb))
    
    a, b = (a, b) if a < b else (b, a)
    
    x = a + 0.5 * abs(a - b)
    y = f(x)
    
    return i, x, y


# Algorithm 3.3: p.41
def golden_section_search(f, x, epsilon = 1E-6, verbose = False):

    a, b = bracket_minimum(f, x)
    
    if verbose:
        print('init:(a:%.4f, b:%.4f)' % (a, b))

    distance = abs(a - b)

    psi = 0.5 * (1. + np.sqrt(5))
    rho = psi - 1.
    d = rho * b + (1. - rho) * a
    yd = f(d)
    
    i = 1
    while distance > epsilon:
        c = rho * a + (1. - rho) * b
        yc = f(c)
        if yc < yd:
            b, d, yd = d, c, yc
        else:
            a, b = b, c
  
        pa, pb = (a, b) if a < b else (b, a)
        
        if verbose:
            print('%d:(a:%.4f, b:%.4f)' % (i, pa, pb))

        distance = abs(pa - pb)
        
        i += 1
        
    a, b = (a, b) if a < b else (b, a)
    
    x = a + 0.5 * abs(a - b)
    y = f(x)
    
    return i, x, y


# algorithm 3.7: p.50
def bracket_sign_change(df, a, b, k = 2.):
    
    if a > b:
        a, b = b, a
    
    center, half_width = 0.5 * (b + a), 0.5 * (b - a)
    
    while df(a) * df(b) > 0:
        
        half_width *= k
        
        a = center - half_width
        b = center + half_width

    return (a, b)
    

# algorithm 3.6: p.50
def bisection(df, init_x, epsilon = 1E-6, verbose = False):
    
    a, b = bracket_sign_change(df, init_x - epsilon, init_x + epsilon)
    
    if verbose:
        print('init:(a:%.4f, b:%.4f)' % (a, b))

    ya, yb = df(a), df(b)

    if ya == 0:
        b = a
    if yb == 0:
        a = b
    
    i = 1
    while b - a > epsilon:
        
        x = 0.5 * (a + b)
        y = df(x)
        
        if y == 0:
            a, b = x, x
        elif y * ya > 0:
            a = x
        else:
            b = x

        if verbose:
            print('step %d - a:%.4f, b:%.4f, y:%.4f, ya:%.4f' % (i, a, b, y, ya))
        
        i += 1
        
        x = a + 0.5 * abs(a - b)
        y = df(x)
    
    return i, x, y

