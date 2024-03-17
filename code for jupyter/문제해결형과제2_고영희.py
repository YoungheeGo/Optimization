# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 13:11:17 2020

@author: koh99
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import GradientDescent as gd
import sympy as sp
########################################
""" Branin Function """ #목적함수
########################################

def draw_branin(levels):

    a = 1
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    
    braninfunction = lambda x1, x2: a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    
    X1, X2 = np.meshgrid(np.linspace(-5., 20., 100), np.linspace(-5.0, 20.0, 100)) #범위가 -5~20사이
    
    Y = braninfunction(X1, X2)
    
    plt.figure(figsize = (8, 8))
    
    plt.contour(X1, X2, Y, np.logspace(-5.0, 20.0, levels, base = 10)) #, cmap = 'gray')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


draw_branin(100)  #등고선 갯수 

# Branin Function
def f(x, a = 1, b = 5.1 / (4 * np.pi ** 2), c = 5 / np.pi, r = 6, s = 10, t = 1 / (8 * np.pi)):
    
    y = a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * sp.cos(x[0]) + s

    return y

x = sp.IndexedBase('x')
gradients = np.array([sp.diff(f(x), x[i]) for i in range(2)])
grads = sp.lambdify(x, gradients, 'numpy')



#### Branin fuction min value 4 ####


# Gradient Descent | ConjugateGradient

alpha = 1E-2 # learning rate

x_=np.array([0.,15.])
gd.GradientDescent(f, grads, x_, alpha)
gd.ConjugateGradient(f, grads, x_, verbose = False)

x_ = np.array([5., 5.])
gd.GradientDescent(f, grads, x_, alpha)
gd.ConjugateGradient(f, grads, x_, verbose = False)

x_=np.array([13.,5.])
gd.GradientDescent(f, grads, x_, alpha)
gd.ConjugateGradient(f, grads, x_, verbose = False)

x_=np.array([20.,17.])
gd.GradientDescent(f, grads, x_, alpha)
gd.ConjugateGradient(f, grads, x_, verbose = False) # error


# Momentum

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
 
x_=np.array([0.,15.])   
momentum(f, grads, x_, alpha = 1E-2, verbose = True)

x_ = np.array([5., 5.])
momentum(f, grads, x_, alpha = 1E-2, verbose = True)

x_=np.array([13.,5.])
momentum(f, grads, x_, alpha = 1E-2, verbose = True)

x_=np.array([20.,17.])
momentum(f, grads, x_, alpha = 1E-2, verbose = True)


# nesterov : 갱신 formula 달라짐
def nesterov(f, grads, x, alpha, beta = 0.9, maxiter = 1000, TOL = 1E-4, verbose = True):
 
    v = -alpha * np.asarray(grads(x))
    x = x + v

    y_prev = f(x)
    
    
    flag = True
    i = 1
    while flag:

        v = beta * v - alpha * np.asarray(grads(x + beta * v)) # Momentum 과의 차별점!
        x = x + v
           
        y = f(x)

        if verbose:
             print('{}: y {:.4f}, x {}'.format(i, y, x))
        
        if abs(y_prev - y) < TOL * (abs(y_prev) + TOL) or i >= maxiter:
            
            flag = False
        
        y_prev = y
        
        i += 1
        
    return i, y, x

x_=np.array([0.,15.])   
nesterov(f, grads, x_, alpha = 1E-2, verbose = True)

x_ = np.array([5., 5.])
nesterov(f, grads, x_, alpha = 1E-2, verbose = True)
# momentum보다 훨씬 빠르고 정확한 값을 찾아줌

x_=np.array([13.,5.])
nesterov(f, grads, x_, alpha = 1E-2, verbose = True)

x_=np.array([20.,17.])
nesterov(f, grads, x_, alpha = 1E-2, verbose = True)


# conjugate 방법과 유사한 결과 -> 반복 수 많음 대신 고정된 학습률 이용
# 스텝사이즈 매번 반복하지 않음


# Adagrad: Adaptive subgradient (2011)
# 변수 별 gradient 특징 반영한 Adagrad

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
        
x_=np.array([0.,15.])  
adagrad(f, grads, x_, alpha = 1E-1, verbose = False)

x_ = np.array([5., 5.])
adagrad(f, grads, x_, alpha = 1E-1, verbose = False)

x_=np.array([13.,5.])
adagrad(f, grads, x_, alpha = 1E-1, verbose = False)

x_=np.array([20.,17.])
adagrad(f, grads, x_, alpha = 1E-1, verbose = False) 

# 계산량 엄청 많아짐 -- 계산 값안정적이기 때문에 다른 경우보다 못한 경우
# x1,x2일때 차이가 많이 날 때 이 경우 좀 더 좋음
# 다른 방법들은 변수 많은 경우에 적합하지 않은데 이건 3개 이상 변수일때 유용



# RMSProp : decay값은 gradient 갑자기 크게 변하는 값 방지

def rmsprop(f, grads, x, alpha, decay = 0.9, epsilon = 1E-7, maxiter = 1000, TOL = 1E-4, verbose = True):

    s = np.zeros_like(x)

    y_prev = f(x)
  
    flag = True
    i = 1
    while flag:
        
        g = np.asarray(grads(x))
        s = decay * s + (1. - decay) * (g * g)  # gradient ^2

        RMS = np.sqrt(s + epsilon) # over-flow 나지 않기 위함
        x = x - alpha * g / RMS
        
        y = f(x) 
        
        if verbose:
             print('{}: y {:.4f}, x {}'.format(i, y, x))
        
        if abs(y_prev - y) < TOL * (abs(y_prev) + TOL) or i >= maxiter:
            
            flag = False
        
        y_prev = y
        
        i += 1
        
    return i, y, x
              


# adagrad와 유사한 방법으로 계산 진행
    
x_=np.array([0.,15.])  
rmsprop(f, grads, x_, alpha = 1.01E-2, verbose = False)

x_ = np.array([5., 5.])
rmsprop(f, grads, x_, alpha = 1.01E-2, verbose = False)

x_=np.array([13.,5.])
rmsprop(f, grads, x_, alpha = 1.01E-2, verbose = False)

x_=np.array([20.,17.])
rmsprop(f, grads, x_, alpha = 1.01E-2, verbose = False)
# 학습률이 작아짐 (최소값 0.3978에 근사하는 학습률로 조정함)



## adadelta (2012)
# s값의 차이를 학습률로 이용
def adadelta(f, grads, x, rho = 0.95, epsilon = 1E-7, maxiter = 1000, TOL = 1E-4, verbose = True):

    s = np.zeros_like(x)
    u = np.zeros_like(x) #초기값

    y_prev = f(x)
  
    flag = True
    i = 1
    while flag:
        
        g = np.asarray(grads(x))
        s = rho * s + (1. - rho) * (g * g) # convex combination
         
        delta = (np.sqrt(u + epsilon) / np.sqrt(s + epsilon)) * g 
         
        u = rho * u + (1. - rho) * delta * delta
        
        x = x - delta  # update formula
        
        y = f(x) 
        
        if verbose:
             print('{}: y {:.4f}, x {}'.format(i, y, x))
        
        if abs(y_prev - y) < TOL * (abs(y_prev) + TOL) or i >= maxiter:
            
            flag = False
        
        y_prev = y
        
        i += 1
        
    return i, y, x
        
# 학습률 없음. rho가 parameter
#adadelta(f, grads, x_, rho = 0.9, verbose = False)
#adadelta(f, grads, x_, rho = 0.5, verbose = False) # 안정적
adadelta(f, grads, x_, rho = 0.03, maxiter = 10000, verbose = False)
# s 값 크게 갱신
# 적절한 하이퍼 파라미터 쉽지 않음

x_=np.array([0.,15.]) 
adadelta(f, grads, x_, rho = 0.03, maxiter = 10000, verbose = False)
 
x_ = np.array([5., 5.])
adadelta(f, grads, x_, rho = 0.03, maxiter = 10000, verbose = False)

x_=np.array([13.,5.])
adadelta(f, grads, x_, rho = 0.03, maxiter = 10000, verbose = False)

x_=np.array([20.,17.])
adadelta(f, grads, x_, rho = 0.03, maxiter = 10000, verbose = False)


## Adam: Adaptive moment estimation (2015)

def adam(f, grads, x, alpha, rho1 = 0.9, rho2 = 0.99, epsilon = 1E-7, maxiter = 1000, TOL = 1E-4, verbose = True):

    k = 0
    v = np.zeros_like(x)
    s = np.zeros_like(x)
 
    y_prev = f(x)
 
    flag = True
    i = 1
    while flag:
        
        g = np.asarray(grads(x))
     
        v = rho1 * v + (1. - rho1) * g
        s = rho2 * s + (1. - rho2) * (g * g)

        k += 1
     
        v_hat = v / (1 - rho1 ** k)
        s_hat = s / (1 - rho2 ** k)
 
        x = x - alpha * (v_hat / np.sqrt(s_hat + epsilon)) 
    
        y = f(x) 
        
        if verbose:
             print('{}: y {:.4f}, x {}'.format(i, y, x))
        
        if abs(y_prev - y) < TOL * (abs(y_prev) + TOL) or i >= maxiter:
            
            flag = False
        
        y_prev = y
        
        i += 1
        
    return i, y, x 
# 단순한 계산 이루어지는 구조
  
x_=np.array([0.,15.])  
adam(f, grads, x_, alpha = 3E-0, verbose = False)

x_ = np.array([5., 5.])
adam(f, grads, x_, alpha = 3E-0, verbose = False)

x_=np.array([13.,5.])
adam(f, grads, x_, alpha = 3E-0, verbose = False)

x_=np.array([20.,17.])
adam(f, grads, x_, alpha = 3E-0, verbose = False)
# 안정적으로 쉽게 찾아짐. 

# conjugate 가 제일 계산 잘 되지만 hessian, 자코비안 행렬 이용하기 때문에 DL 적용 어렵
# 이차 미분 정보 갖는 알고리즘이 최적화에는 더 안정적인 최소값 얻을 수 있음



#### Newton's method : 
# 목적함수는 multivariate 이므로 univaiate 방법(Sevant 등)은 생략함


hessian=np.asarray([[sp.diff(gradients[i],x[j]) for i in range(2)] for j in range(2)])
hess=sp.lambdify(x,hessian,'numpy')


#x_ = np.array([5., 5.])
#x_-np.dot(np.linalg.inv(hess(x_)),grads(x_)) #어떤 초기값을 입력해도 minimum값에 도출
# 최소 제곱 추정량

## 다변량 Newtom
def newton(f, grads, hessian, x, maxiter = 1000, TOL = 1E-4, verbose = True):

    y_prev = f(x)
  
    flag = True
    i = 1
    while flag:
    
        x = x - np.dot(np.linalg.inv(hess(x)), grads(x)) # update formula

        y = f(x) 
        
        if verbose:
             print('{}: y {:.4f}, x {}'.format(i, y, x))
        
        if abs(y_prev - y) < TOL * (abs(y_prev) + TOL) or i >= maxiter:
            
            flag = False
        
        y_prev = y
        
        i += 1
        
    return i, y, x

# 차원의 수 커지면 계산 힘들어짐 : np.linalg.inv
x_ = np.array([-2., 10.])
newton(f, grads, hess, x_)

x_ = np.array([5., 5.])
newton(f, grads, hess, x_)    


x_ = np.array([9., 8.])
newton(f, grads, hess, x_)

x_ = np.array([16., 15.])
newton(f, grads, hess, x_)


## Noisy descent

def NoisyDescent(f, grads, x, alpha, maxiter = 1000, TOL = 1E-4, verbose = True):
    
    y_prev = f(x)
    
    flag = True
    i = 1
    while flag:

        g = np.asarray(grads(x))
        g = - g / np.sqrt(np.dot(g, g)) # Normalizing
        
        noise = np.random.normal(loc = 0, scale = 1. / i , size = g.shape)

        x = x + alpha * g + noise

        y = f(x)

        if verbose:
            print('{}: y {:.4f}, x {}'.format(i, y, x))
        
        if abs(y_prev - y) < TOL * (abs(y_prev) + TOL) or i >= maxiter:
            
            flag = False
        
        y_prev = y
        
        i += 1
        
    return i, y, x

x_ = np.array([-2., 10.])
NoisyDescent(f, grads, x_, alpha = 0.01)
# 처음 시작시 정규 분포의 난수값이 최소값과 다른 방향이면 못 찾을 수도 있음
# 난수는 시행시 달라짐 -> 여러번 수행

y_best=0.3978
x_best=[-3., 12.]
x_ = np.array([0.,15.])
for i in range(10):
    i,y,x =NoisyDescent(f,grads,x_,alpha=0.01,verbose=False)
    if y<y_best:
        y_best,x_best=y,x
print(y_best,x_best)

y_best=0.3978
x_best=[3., 2.]
x_ = np.array([5.,5.])
for i in range(10):
    i,y,x =NoisyDescent(f,grads,x_,alpha=0.01,verbose=False)
    if y<y_best:
        y_best,x_best=y,x
print(y_best,x_best)

y_best=0.3978
x_best=[9., 2.]
x_ = np.array([13.,5.])
for i in range(10):
    i,y,x =NoisyDescent(f,grads,x_,alpha=0.01,verbose=False)
    if y<y_best:
        y_best,x_best=y,x
print(y_best,x_best)

y_best=0.3978
x_best=[15., 12.]
x_ = np.array([20.,17.])
for i in range(10):
    i,y,x =NoisyDescent(f,grads,x_,alpha=0.01,verbose=False)
    if y<y_best:
        y_best,x_best=y,x
print(y_best,x_best)


# 확률적인 방법은 계산 1번 수행이 아니라 난수 이용하기 때문에 여러번 반복 계산 시키고 
# 그 중 목적함수 값이 가장 작은 값을 최적화 문제의 근사 해!!

# NoisyDescent 문제는 Saddle point 나 local minimum 같은 문제 해결가능 
# 확률 적 방법이기 때문에 동일한 NoisyDescent 여러번 수행되야함
# Nueral Net, 기계학습에서 많은 사용!


## Simulated annealing

def SimulatedAnnealing(f, x, t, schedule, decay = 0.9, maxiter = 1000, TOL = 1E-4):

    y = f(x)
    y_best, x_best = y, x
    t0 = t

    flag = True
    i = 1
    while flag:    

        x_ = x + np.random.normal(size = len(x)) # 난수 생성 
        y_ = f(x_)
        dy = y_ - y 

        # clip : 값의 범위 제한
        t = float(max(t, 0.1))

        if dy < 0:
            dy = float(max(-10, dy))
        else:
            dy = float(min(10, dy))

        if y_ <= 0 or np.random.uniform() < min(np.exp(-dy / t), 1):
            x, y = x_, y_
            
        if y_ < y_best:
            y_best, x_best = y_, x_

        if schedule == 'logarithmic':
            t = t * np.log(2) / np.log(i + 1)
        elif schedule == 'exponential':
            t = decay * t
        elif schedule == 'fast':
            t = t0 / i            

        if i >= maxiter:
            
            flag = False
        
        i += 1
        
    return i, y_best, x_best


# Gradient 필요 x. 대신 온도 필요 어떨게 바꿔줘야하는지 schedule 필요
# 종료 조건 위한 아규먼트

t = 1

x_=np.array([0.,15.])  
SimulatedAnnealing(f, x_, t, 'logarithmic')
SimulatedAnnealing(f, x_, t, 'exponential')
SimulatedAnnealing(f, x_, t, 'fast')

x_ = np.array([5., 5.])
SimulatedAnnealing(f, x_, t, 'logarithmic')
SimulatedAnnealing(f, x_, t, 'exponential')
SimulatedAnnealing(f, x_, t, 'fast')

x_=np.array([12.,5.])
SimulatedAnnealing(f, x_, t, 'logarithmic')
SimulatedAnnealing(f, x_, t, 'exponential')
SimulatedAnnealing(f, x_, t, 'fast')

x_=np.array([20.,17.])
SimulatedAnnealing(f, x_, t, 'logarithmic')
SimulatedAnnealing(f, x_, t, 'exponential')
SimulatedAnnealing(f, x_, t, 'fast')

