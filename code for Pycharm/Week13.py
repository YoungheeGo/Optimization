##############################################################
"""                       Week 13                          """
##############################################################

""" 기말평가 Hint 
1) 교재 p.17 Exercise 1.2 # 최소값
2) 교재 p.33 Exercise 2.5 # computational graph
3) 교재 p.51 Exercise 3.4 # bisection algorithm
4) 교재 p.85 Exercise 5.7 # conjugate gradient descent
   .....
"""

import numpy as np

def michalewicz(x, m = 10):
    
    return - sum([np.sin(v) * np.sin(((i+1) * (v ** 2)) / np.pi) ** (2 * m) for i, v in enumerate(x)])


michalewicz([2.2, 1.57])

POP_SIZE = 10
DIM = 2

population = np.random.uniform(low = 0, high = 4, size = (POP_SIZE, DIM))
collection = [michalewicz(ind) for ind in population]


def selection(f, population, k):

    collection = [f(ind) for ind in population]
    pool = np.argsort(collection)    

    return [population[p] for i, p in enumerate(pool) if i < k]

new_population = selection(michalewicz, population, 4)


def parents(population):
    
    parent = list()
    for _ in range(POP_SIZE):
        idx = np.random.randint(low = 0, high = len(population), size = len(population[0]))
        parent.append([population[idx[0]], population[idx[1]]])
    
    return parent


parent = parents(new_population)    


def crossover(parents, lambda_ = 0.5):
    
    return  [(1. - lambda_) * p[0] + lambda_ * p[1] for p in parents]


child = crossover(parent)


def mutation(child, sigma):
    
    return child + np.random.normal(loc = 0, scale = sigma, size = (len(child), len(child[0])))


mutation(child, 1. / len(child))


def genetic_algorithm(f, pop_size, dim, k, maxiter = 100):

    population = np.random.uniform(low = 0, high = 4, size = (pop_size, dim))
    collection = [f(ind) for ind in population]
    
    flag = True
    i = 1
    
    while flag:
       
        population = selection(f, population, k)
        parent = parents(population)
        child = crossover(parent)

        population = mutation(child, 1. / len(child))

        collection = [f(ind) for ind in population]

        if i > maxiter:
            break
        
        i += 1

    pool = np.argsort(collection)    
    
    return collection[pool[0]], population[pool[0]]
   

POP_SIZE = 50
DIM = 2
k = 10

genetic_algorithm(michalewicz, POP_SIZE, DIM, k = 10)        
        
