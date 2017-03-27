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
                error = int(train_y[i,0])-output
                weights = weights + alpha*train_x[i,:].transpose()*error
        # smooth SGD
        elif opts['optimizeType'] == 'smoothStocGradDescent':
            samples_index = range(samples_num)
            for i in samples_index:
                # alpha initialization
                alpha = 4.0 / (1.0 + iterTime + i) + 0.01
                randIndex =int(np.random.uniform(0,len(samples_index)))
                output = sigmoid(train_x[randIndex,:]*weights)
                error = train_y[randIndex,0]-output
                weights = weights + alpha*train_x[randIndex,:].transpose()*error
                # during one iteration
                del(samples_index[randIndex])
        else:
            raise NameError('Not support optimize method type!')

    print "Training complete. Took %f s" %(time.time()-start_time)
    return weights

# test the trained LR model given test set
def testLR(weights,test_x,test_y):
    sample_num, feature_num = test_x.shape
    matchCount = 0
    for i in range(sample_num):
        predict = sigmoid(test_x[i,:]*weights)[0,0]>0.5
        if predict == bool(test_y[i,0]):
            matchCount += 1
    accuracy = float(matchCount)/sample_num
    return accuracy

# plot
def showLR(weights,train_x,train_y):
    sample_num,feature_num = train_x.shape
    if feature_num != 3:
        print "sorry! dimension is not 2!"
        return 1
    # draw samples
    for i in range(sample_num):
        if int(train_x[i,0] == 0):
            plt.plot(train_x[i,1],train_x[i,2],'or')
        elif int(train_y[i,0] == 1):
            plt.plot(train_x[i,1],train_x[i,2],'ob')
    # draw the classify line
    min_x = min(train_x[:,1])[0,0]
    max_x = max(train_x[:,1])[0,0]
    # convert matrix to array
    weights = weights.getA()
    y_min_x = float(-weights[0] - weights[1]*min_x)/weights[2]
    y_max_x = float(-weights[0] - weights[1]*max_x)/weights[2]
    plt.plot([min_x,max_x],[y_min_x,y_max_x],'-g')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()






