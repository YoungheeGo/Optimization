##############################################################
""" Week 5 """
##############################################################

import numpy as np

from IPython.display import Image
Image("./fig1.png", width = 640)#, height = 300)


# Algorithm 3.1: p.36
def bracket_minimum(f, x, s = 1E-2, k = 2.0):
    
    a, ya = x, f(x)
    b, yb = a + s, f(a + s)
    
    print('init: (a:%.4f, b:%.4f) (ya:%.4f, yb:%.4f)' % (a, b, ya, yb))
     
    if yb > ya:
        a, b = b, a
        ya, yb = yb, ya
        s = -s

    while True:
        c, yc = b + s, f(b + s)
        print('step: (a:%.4f, b:%.4f, c:%.4f) (ya:%.4f, yb:%.4f, yc:%.4f)' % (a, b, c, ya, yb, yc))
        
        if yc > yb:
            return (a, c) if a < c else (c, a)
        else:
            a, ya, b, yb = b, yb, c, yc
            s *= k


# Algorithm 3.3: p.41
def golden_section_search(f, x, epsilon = 1E-6):

    a, b = bracket_minimum(f, x)
    print('init:(a:%.4f, b:%.4f)' % (a, b))

    distance = abs(a - b)

    psi = 0.5 * (1. + np.sqrt(5))
    rho = psi ** (- 1)
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
        print('%d:(a:%.4f, b:%.4f)' % (i, pa, pb))

        distance = abs(pa - pb)
        
        i += 1
        
    a, b = (a, b) if a < b else (b, a)
    
    x = 0.5 * (a + b)
    y = f(x)
    
    return x, y

    
# Algorithm 4.1: p.54
def line_search(f, x, d):
    
    def obj(alpha):
        
        return f(x + alpha * d)
    
    #a, b = bracket_minimu(ojb, 0.)
    
    alpha, _ = golden_section_search(obj, 0.)

    return alpha, x + alpha * d


# Example 4.1: p.55
def f(x):
    
    y = np.sin(x[0] * x[1]) + np.exp(x[1] + x[2]) - x[2]
    
    return y

x = np.asarray([1, 2, 3])
d = np.asarray([0, -1, -1])

line_search(f, x, d)


from IPython.display import Image
Image("./fig3.png", width = 640)#, height = 300)

# Algorithm 4.2: p.56
def backtracking_line_search(f, gradient, x, d, alpha, p = 0.5, beta = 1E-4):
    
    y, g = f(x), gradient
    
    i = 1
    while f(x + alpha * d) > y + beta * alpha * np.dot(g, d): # g.T @ d
    
        alpha *= p
        
        print('%d: alpha = %.4f' % (i, alpha))
        
        i +=1
    
    return alpha


# Example 4.2: p. 58
def f(x):
    
    y = x[0] ** 2 + x[0] * x[1] + x[1] ** 2
    
    return y


def pdf0(x):
    
    return 2 * x[0] + x[1]


def pdf1(x):
    
    return 2 * x[1] + x[0]


x = np.asarray([1, 2])
d = np.asarray([-1, -1])
gradient = np.asarray([pdf0(x), pdf1(x)])

alpha = 10

alpha = backtracking_line_search(f, gradient, x, d, alpha)

x = x + alpha * d
print(x)

