###############################################
"""                   Week 7                """
###############################################

# 1. Symbolic differenciation

# 2. Bactracking line search / Strong backtracking with Symbolic differencitaion

# 3. Local descent : multivariate

# 4. 문제해결 과제 : univariate

""" 
sympy 설치

conda install sympy
"""

import numpy as np

# Symbolic differenciation

import sympy as sp

# Example 3.1 : univariate

x = sp.Symbol('x')

f = sp.exp(x - 2) - x

df = f.diff(x)

print(df)

derv = sp.lambdify(x, df, 'numpy') # x, df: symbol

print(derv(1))
print(derv(2))
print(derv(3))


# Example 4.2: p. 58

def f(x):

    y = x[0] ** 2 + x[0] * x[1] + x[1] ** 2

    return y


#multivariate
x = sp.IndexedBase('x')

sp.diff(f(x), x[0])
sp.diff(f(x), x[1])

[sp.diff(f(x), x[i]) for i in range(2)]

gradients = np.array([sp.diff(f(x), x[i]) for i in range(2)])

grads = sp.lambdify(x, gradients, 'numpy')

x_ = [1, 2]

print(grads(x_))



# Example 4.1 : multivariate

def f(x):
    
    return sp.sin(x[0] * x[1]) + sp.exp(x[1] + x[2]) - x[2]

x = sp.IndexedBase('x')

gradients = np.array([sp.diff(f(x), x[i]) for i in range(3)])

grads = sp.lambdify(x, gradients, 'numpy')

x_ = [1, 2, 3]
print(grads(x_))


# Algorithm 4.2: p.56
def backtracking_line_search(f, grads, x, d, alpha = 10, p = 0.5, beta = 1E-4):
    
    y, g = f(x), grads(x)
    
    i = 1
    while f(x + alpha * d) > y + beta * alpha * np.dot(g, d): # g.T @ d
        
        alpha *= p

        print('%d: alpha = %.4f' % (i, alpha))
        
        i += 1
        
    return alpha


# Example 4.2: p. 58

def f(x):
    
    y = x[0] ** 2 + x[0] * x[1] + x[1] ** 2
    
    return y

x = sp.IndexedBase('x')

gradients = np.array([sp.diff(f(x), x[i]) for i in range(2)])

grads = sp.lambdify(x, gradients, 'numpy')

x_ = np.array([1, 2])
d_ = np.array([-1, -1])

alpha = backtracking_line_search(f, grads, x_, d_)
print(alpha)


# Local descent algorithm
# 1. x(k) 종료조건 확인
# 2. descent direction d(k) 결정
# 3. alpha(k) 결정
# 4. x(k+1) = x(k) + alpha(k) * d(k)

# x = x + alpha * (g * d) : d = [-1]

x_ = np.array([1, 2])
d_ = -1 * np.array(grads(x_)) # local model: descent direction

alpha = backtracking_line_search(f, grads, x_, d_)

print(alpha, x_, f(x_))


# Termination conditions

# 1. abstol: abs(y - y_prev) < 1E-8

# 2. reltol: abs(y - y_prev) < 1E-8 * (abs(y_prev) + 1E-8)

# 3. maximum iteration

y_prev = f(x_)

maxiter = 100

flag = True
i = 1
while flag:
    
    x_ = x_ + alpha * d_
    d_ = -1 * np.array(grads(x_))
     
    alpha = backtracking_line_search(f, grads, x_, d_)

    y_ = f(x_)
    
    diff = np.abs(y_ - y_prev)
    print(i, alpha, x_, f(x_), diff)
    
    #if diff < 1E-8:
    #    flag = False
    
    #if diff < 1E-8 * (np.abs(y_prev) + 1E-8):
    #    flag = False
    if i > maxiter: 
        flag = False
        
    y_prev = y_
    
    i += 1 
    
    
# local descent with backtracking line search
    
 
def local_descent_backtracking(f, grads, x_, alpha = 10, TOL = 1E-8):

    d_ = -1 * np.array(grads(x_))
    
    alpha = backtracking_line_search(f, grads, x_, d_, alpha)
    
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
        print(i, alpha, x_, f(x_), diff)

        if diff < TOL * (abs(y_prev) + TOL):
            flag = False
   
        y_prev = y_
        
        i += 1

    return i, x_


x_ = np.array([1, 2])
local_descent_backtracking(f, grads, x_)


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

            
def local_descent_strong_backtracking(f, grads, x_, alpha = 10, TOL = 1E-8):

    d_ = -1 * np.array(grads(x_))
    
    alpha = strong_backtracking(f, grads, x_, d_, alpha)
    
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
        print(i, alpha, x_, f(x_), diff)

        if diff < TOL * (abs(y_prev) + TOL):
            flag = False
    
        y_prev = y_
        i += 1
    
    return i, x_

x_ = np.array([1, 2])
local_descent_strong_backtracking(f, grads, x_)


from IPython.display import Image
Image("./fig7.png", width = 640)

  
# Rosenbrock
def f(x, a = 1, b = 5):
    y = (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2    
    return y


x = sp.IndexedBase('x')
gradients = np.array([sp.diff(f(x), x[i]) for i in range(2)])
grads = sp.lambdify(x, gradients, 'numpy')

x_ = np.array([-2, 2])

local_descent_backtracking(f, grads, x_, 1)

local_descent_strong_backtracking(f, grads, x_, 1)


from IPython.display import Image
Image("./fig8.png", width = 640)


# 문제해결과제
def f(x):
    y = x[0] ** 5 - 8 * x[0] **3 + 10 * x[0] + 6
    return y

x = sp.IndexedBase('x')
gradients = np.array([sp.diff(f(x), x[i]) for i in range(1)])
grads = sp.lambdify(x, gradients, 'numpy')


x_ = np.array([-2])

local_descent_backtracking(f, grads, x_, 0.1)

local_descent_strong_backtracking(f, grads, x_, 1)


# 8 주차
# 3개 함수 : 종료조건에 따른 비교(backtraking line search, strong backtraking)
# 문제해결 함수 : alpha, x_, 종료조건

# Keystroke dynamics 자료 수집 협조 요청^^



