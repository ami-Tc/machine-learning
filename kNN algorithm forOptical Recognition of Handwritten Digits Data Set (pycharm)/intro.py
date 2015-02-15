from numpy import *
from operator import itemgetter
from os import listdir

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDiffMatSum = sqDiffMat.sum(axis=1)
    distances = sqDiffMatSum**0.5
    distSort = distances.argsort()
    classCount = {}

    for i in range (k) :
        voteLabel = labels[distSort[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    sortedClassCount = sorted(classCount.iteritems(), key=itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def imageToMatrix(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        fileStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i + j] = fileStr[j]
    return returnVect

def hdReco():
    targetDir = listdir('trainingDigits')
    m = len(targetDir)
    dataSet = zeros((m, 1024))
    hwLabels = []

    for i in range(m):
        dataSet[i, :] = imageToMatrix('trainingDigits/%s' %targetDir[i])
        fileNameStr = targetDir[i].split('.')[0]
        filename = fileNameStr.split('_')[0]
        hwLabels.append(int(filename))

    testDir = listdir('testDigits')
    mTest = len(testDir)
    testVec = zeros((1,1024))
    errorCount = 0.0

    for i in range(mTest):
        testVec = imageToMatrix('testDigits/%s' %testDir[i])
        fileNameStr = testDir[i].split('.')[0]
        filename = int(fileNameStr.split('_')[0])
        retClass = classify0(testVec, dataSet, hwLabels, 50)
        print "classifier returned value : %d, actual value : %d" %(retClass, filename)
        if (retClass != filename):
            errorCount += 1

    print "error count is %d" %errorCount
    print "error rate : %f" %(errorCount/float(mTest))


hdReco()