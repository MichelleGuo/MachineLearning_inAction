import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# sigmoid function
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def trainLR(train_x,train_y,opts):
    # training time
    start_time = time.time()
    # train set is a m*n matrix, each row stands for one sample
    # train_y is mat datatype too, each row is the corresponding label
    samples_num,features_num = train_x.shape
    alpha = opts['alpha']
    maxIter = opts['maxIter']
    weights = np.ones((features_num,1))

    # SGD optimal
    for iterTime in range(maxIter):
        # gradient descent algorithm
        if opts['optimizeType'] == 'gradientDescent':
            output = sigmoid(train_x*weights)
            error = train_y-output
            weights = weights + alpha*train_x.transpose()*error
        # SGD
        elif opts['optimizeType'] == 'stocGradDescent':
            for i in range(samples_num):
                # SGD works for one sample
                output = sigmoid(train_x[i,:]*weights)
                error = train_y[i,0]-output
                weights = weights + alpha*train_x[i,:].transpose()*error
        # smooth SGD
        elif opts['optimizeType'] == 'smoothStocGradDescent':
            samples_count = range(samples_num)
            for i in samples_count:
                # alpha initialization
                alpha = 4.0 / (1.0 + iterTime + i) + 0.01


