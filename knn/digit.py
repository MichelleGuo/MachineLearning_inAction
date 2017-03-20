import numpy as np
import operator
import math


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



