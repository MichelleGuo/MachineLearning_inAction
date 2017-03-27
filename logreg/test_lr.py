import numpy as np
from lr import *

def loadData():
    train_x = []
    train_y = []
    fileIn = open('testSet.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split()
        train_x.append([1.0,float(lineArr[0]),float(lineArr[1])])
        train_y.append(lineArr[2])
    return np.mat(train_x),np.mat(train_y).transpose()

print "step 1: loading data..."
train_x,train_y = loadData()
test_x = train_x
test_y = train_y

print "step 2: start training..."
# learning rate,iteration times,optimize method
opt = {'alpha':0.01,'maxIter':20,'optimizeType':'smoothStocGradDescent'}
optimal_weights = trainLR(train_x,train_y,opt)

print "step 3: start testing..."
accuracy = testLR(optimal_weights,test_x,test_y)

print "step 4: show the result..."
print "The accuracy is %.3f%%" %(accuracy*100)
showLR(optimal_weights,train_x,train_y)
