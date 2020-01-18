from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX, dataSet, labels, k):
    """
    :param inX: 未知类别属性点 ，是array类型
    :param dataSet: 训练数据集中的属性点，是array类型
    :param labels: 训练数据集中的标签，是list类型
    :param k: 选择与inX距离最小的前k个点
    :return: 返回前k个点中同一类别最多的那一类标签
    """
    dataSetSize = dataSet.shape[0]       #array_name.shape返回数组的行列(m,n)元组
    diffMat = tile(inX,(dataSetSize,1)) - dataSet       #得到未知类别点与所有已知类别点的坐标差
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)                #axis=1表示按行求和，=0表示按列求和
    distances = sqDistances**0.5
    sortedDisIndicies = distances.argsort()            #返回递增排序的索引，类型为array
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDisIndicies[i]]      #选择与inX距离最小的k个点的类别
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  #查找字典classCount中是否存在关键值voteIlabel，没有返回0
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #key=operator.itemgetter(1)表示根据第二个域（类别数量）逆序排序,classCount.items()返回字典的键值对列表
    return sortedClassCount[0][0]

def img2vector(filename):
    """
    :param filename:The name of open file
    :return: a vector which size is 1*1024
    """
    returnVect = zeros((1,1024))    #默认情况下，zeros创建的numpy数组中的元素是浮点型的
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()   #读取文件fr的一整行，包括‘\n’
        for j in range(32):
            returnVect[0,i*32+j] = int(lineStr[j])
    fr.close()
    return returnVect

def hanwritingClassTest():
    hwlabel = []
    trianingFileList = listdir('trainingDigits')     #trainingFileList是字符串数组，保存的是trainingDigits中的文件名
    m = len(trianingFileList)
    trainingMat = zeros((m,1024))             #该矩阵的每一行保存一个文本存储的数字的转换的向量
    for i in range(m):
        fileNameStr = trianingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = fileStr.split('_')[0]      #可以从文本名解析出分类数字
        hwlabel.append(classNumStr)   #将trainingDigits中每个文本所代表的数字标签保存在列表hwlabel中
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)

    testFileList = listdir('testDigits')
    mtest = len(testFileList)
    errorCount = 0.0
    for i in range(mtest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwlabel,3) #返回的classifierResult是字符串类型
        print('The classifier come back with: %d, The real answer is: %d' % (int(classifierResult),classNumStr))
        if (int(classifierResult) != classNumStr):
            errorCount+=1.0
    print('\nThe total number of errors is: %d' %errorCount)
    print('\nThe total error rate is: %f' %(errorCount/float(mtest)))




