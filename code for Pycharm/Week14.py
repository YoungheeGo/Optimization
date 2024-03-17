##############################################################
"""                       Week 14                          """
##############################################################

""" Stochastic Gradient Descent """


import numpy as np

from sklearn import datasets

dataset = datasets.load_diabetes()
print(dataset.DESCR)

data = dataset.data
print(data.shape)

target = dataset.target

# shuffling
shuffle_idx = np.random.choice(data.shape[0], data.shape[0], replace = False)

data[shuffle_idx]
target[shuffle_idx]


mini_batch_size = 40
num_minibatch = data.shape[0] // mini_batch_size
print(num_minibatch)

for m in range(num_minibatch):
    mini_batch = data[(m * mini_batch_size):((m + 1) * mini_batch_size)]
    mini_X = mini_batch[:, :-1]
    mini_y = mini_batch[:, -1]
    print(mini_X.shape, mini_y.shape)

    
theta = np.random.uniform(-1., 1. , size = data.shape[1])
print(theta)

alpha = 0.01


for m in range(num_minibatch):
    mini_X = data[(m * mini_batch_size):((m + 1) * mini_batch_size)]
    mini_y = target[(m * mini_batch_size):((m + 1) * mini_batch_size)]

    hat_y = np.dot(mini_X, theta)
    theta = theta - alpha * np.dot((hat_y - mini_y), mini_X)
    print(theta)
   
    
alpha = 0.01
epoch = 10
theta = np.random.uniform(-1., 1. , size = data.shape[1])

for iepoch in range(epoch):
    
    shuffle_idx = np.random.choice(data.shape[0], data.shape[0], replace = False)

    data = data[shuffle_idx]
    target = target[shuffle_idx]

    loss = 0.
    for m in range(num_minibatch):
        mini_X = data[(m * mini_batch_size):((m + 1) * mini_batch_size)]
        mini_y = target[(m * mini_batch_size):((m + 1) * mini_batch_size)]
    
        hat_y = np.dot(mini_X, theta)
        
        theta = theta - alpha * np.dot((hat_y - mini_y), mini_X)

        loss += np.mean((hat_y - mini_y) ** 2)

    print(loss / mini_batch_size)
    #print(theta)  

def sgd(data, target, alpha, minibatch_size, epoch, verbose = False):

    theta = np.random.uniform(-1., 1. , size = data.shape[1])
    num_minibatch = data.shape[0] // minibatch_size
    
    for iepoch in range(epoch):
        
        shuffle_idx = np.random.choice(data.shape[0], data.shape[0], replace = False)
    
        data = data[shuffle_idx]
        target = target[shuffle_idx]
    
        loss = 0.
        for m in range(num_minibatch):
            mini_X = data[(m * minibatch_size):((m + 1) * minibatch_size)]
            mini_y = target[(m * minibatch_size):((m + 1) * minibatch_size)]
        
            hat_y = np.dot(mini_X, theta)
            
            theta = theta - alpha * np.dot((hat_y - mini_y), mini_X)
            
            loss += np.mean((hat_y - mini_y) ** 2)
    
        if verbose:
            print('epoch %d: loss = %.4f' % ((iepoch + 1), loss / minibatch_size))
        
    return loss / minibatch_size, theta


data = dataset.data
target = dataset.target

alpha = 0.1 
minibatch_size = 400 #data.shape[0]
epoch = 1000
sgd(data, target, alpha, minibatch_size, epoch, verbose = True)
    