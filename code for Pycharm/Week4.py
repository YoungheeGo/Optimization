##############################################################
""" Week 4 """
##############################################################

import numpy as np

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

from IPython.display import Image
Image("./golden_section.png", width = 640)#, height = 300)

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

        distance = abs(a - b)
 
        i += 1
        
    a, b = (a, b) if a < b else (b, a)
   
    x = 0.5 * (a + b)
    y = f(x)
    
    return x, y


"""
def f(x):
    return 0.5 * x ** 2 - x

def df(x):
    return x - 1
"""

f = lambda x: 0.5 * x ** 2 - x
df = lambda x: x-1

import matplotlib.pyplot as plt
#from matplotlib import pyplot as plt

x = np.arange(0., 2., 1E-2)

plt.title('f(x) = (x ** 2) / 2 - x')
plt.plot(x, f(x))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

plt.title("f`(x) = x - 1")
plt.plot(x, df(x))
plt.axhline(y = 0, color = 'orange')
plt.axvline(x = 1, color = 'orange')
plt.show()


golden_section_search(f, 0)


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

bracket_sign_change(df, 0, 0.1)


# algorithm 3.6: p.50
def bisection(df, x, epsilon = 1E-6):
    
    a, b = bracket_sign_change(df, x - epsilon, x + epsilon)
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

        print('step %d - a:%.4f, b:%.4f, y:%.4f, ya:%.4f' % (i, a, b, y, ya))
        
        i += 1
  
        x = 0.5 * (a + b)
        y = df(x)
  
    return x, y


bisection(df, 0)











