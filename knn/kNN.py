from numpy import *
import operator
import os
import numpy as np
from os import listdir

def createDataSet():
    group=array(([1.0,1.1], [1.0,1.0], [0,0], [0,0.1]))
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    # number of row
    dataSetSize = dataSet.shape[0]
    # inX is replicated in a grid of dataSetSize rows and 1 column
    diffMat = tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()

    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # 字典的get方法，查找classCount中是否包含voteIlabel，是则返回该值，不是则返回defValue，这里是0
        # 其实这也就是计算K临近点中出现的类别的频率，以次数体现
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        # 对字典中的类别出现次数进行排序，classCount中存储的是 key-value，其中key就是label，value就是出现的次数
        # 所以key=operator.itemgetter(1)选中的是 value，也就是对次数进行排序
        sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
        #sortedClassCount[0][0]也就是排序后的次数最大的那个label
        return sortedClassCount[0][0]


def file2matrix(filename):
        fr=open(filename)
        #读取文件
        arrayOLines=fr.readlines()
        #文件行数
        numberOfLines=len(arrayOLines)
        #创建全0矩阵
        returnMat=zeros((numberOfLines,3))
        #标签向量
        classLabelVector=[]
        index=0
        #遍历每一行，提取数据
        # 将每一行回车符截取掉，去掉前后空格
        # strip() 方法用于移除字符串头尾指定的字符（默认为空格）
        # strip()方法语法：str.strip([chars]);
        # 参数chars -- 移除字符串头尾指定的字符。
        for line in arrayOLines:
                line=line.strip();
                # 使用tab字符'\t'将上一步得到的整行数据分割成一个元素列表
                listFromLine=line.split('\t')
                #前三列为属性信息
                returnMat[index,:]=listFromLine[0:3]
                #最后一列为标签信息
                classLabelVector.append(int(listFromLine[-1]))
                index +=1
        return returnMat,classLabelVector

# 归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = minVals - maxVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

# 首先使用了file2matrix和autoNorm函数从文件中读取数据并将其转换为归一化特征值
# 接着计算测试向量的数量，此步决定了normMat向量中哪些数据用于测试
# 哪些数据用于分类器的训练样本；然后将这两部分数据输入到原始kNN分类器函数classifyO
# 最后，函数计算错误率并输出结果.注意此处我们使用原始分类器
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('./datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with:%d, the real answer is: %d " %(classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):errorCount+=1.0
    print("the total error rate is : %f" %(errorCount/float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream=float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('./datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("you will probably like this person:",resultList[classifierResult-1])



# 将图像转换为向量：该函数创建1*1024的numpy数组
# 然后打开给定的文件，循环读出文件的前32行
# 并将每行的头32个字符值存储在numpy数组中，最后返回数组
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
        return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('./trainingDigits')
    m=len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('trainingDigits/%s' %fileNameStr)
    testFileList = listdir('./testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector('./testDigits/%s' %fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with:%d,the real answer is %d" %(classifierResult,classNumStr))
        if(classifierResult != classNumStr):errorCount+=1.0

    print("\nthe total number of errors is:%d" %errorCount)
    print("\nthe total error rate is: %f" %(errorCount/float(mTest)))