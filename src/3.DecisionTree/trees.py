#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from math import log
import operator
import treePlotter as dtPlot
from collections import Counter
import matplotlib
# ## 决策树的一般流程
#
# - 1.收集数据：可以使用任何方法
# - 2.准备数据：树构造算法只适用于标称数据类型，因此数值型数据比徐离散化。
# - 3.分析数据：可以使用任何方法，构造树完成后，我们应该检查图形是否符合预期。
# - 4.NativeBayes.训练算法：构造树的数据结构。
# - 5.测试算法：使用经验树计算错误率。
# - 6.使用算法：此步骤可以适用于任何监督学习算法，而使用决策树可以更好地理解数据的内在含义



# 程序清单3-1 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] +=1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(float(prob), 2)
    return shannonEnt


# - 1.收集数据：可以使用任何方法
# 构造数据
def createDataSet():
    dataSet = [[1,1,'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels


# - 2.准备数据：树构造算法只适用于标称数据类型，因此数值型数据比徐离散化。
# 划分数据集
# 程序清单3-2 按照给定的特征划分数据集
def splitDataSet(dataSet,axis,value):
    """
          Desc:
              按照给定的特征划分数据集
          Args:
              dataSet -- 带划分的数据集
              axis -- 划分数据集的特征
              value -- 特征的返回值
          Returns:
              retDataSet -- 特征数据集
          调用方式
          import kNN
          group, labels = kNN.createDataSet()
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 程序清单3-3 选择最好饿数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prop = len(subDataSet)/float(len(dataSet))
            newEntropy += prop * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
      #  print('infoGain=', infoGain, 'bestFeature=', i, baseEntropy, newEntropy)
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter, reverse=True)
    return sortedClassCount[0][0]


#程序清单3-4.NativeBayes 创建书的函数代码
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    print("classList = ", classList, "classListCounts = ", len(classList))
    # `count()` 方法用于统计某个元素在列表中出现的次数。
    # 递归结束条件 当获取的分类的 labels 都是一样的，递归结束
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # dataSet 每一个数据集，只剩下 目标类（labels）时，返回labels 中出现次数多的 label
    # 此时，对所有 feature（特征向量）已经划分完了，
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 最佳 特征向量
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 最佳特征向量的标签集
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    #print(myTree)
    del(labels[bestFeat])
    featValues = [data[bestFeat] for data in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
        print(myTree)
    return myTree


# 测试算法：使用决策树进行分类
# 程序清单3-8 使用
def classify(inputTree, featLabels,testVec):
    #
    #.TypeError: ‘dict_keys’ object does not support indexing
    # 这个问题是python版本的问题
    # 1
    # #如果使用的是python2
    # firstStr = myTree.keys()[0]
    # #LZ使用的是python3
    # firstSides = list(myTree.keys())
    # firstStr = firstSides[0]
    # ---------------------
    # firstStr = inputTree.keys()[0]  该写法是错误的 在python3 下
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


# 使用算法：决策树的存储
def storeTree(inputTree, fileName):
    import pickle
    fw = open(fileName, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(fileName):
    import pickle
    fr = open(fileName)
    return pickle.load(fr)


# 测试实例
def test():
    dataSet, labels = createDataSet()
    print(dataSet)
    print(labels)
   # print(chooseBestFeatureToSplit(dataSet))
    print(createTree(dataSet, labels))


def ContactLensesTest():
    """
    Desc:
        预测隐形眼镜的测试代码，并将结果画出来
    Args:
        none
    Returns:
        none
    """

    # 加载隐形眼镜相关的 文本文件 数据
    fr = open('../db/3.DecisionTree/lenses.txt')
    # 解析数据，获得 features 数据
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 得到数据的对应的 Labels
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 使用上面的创建决策树的代码，构造预测隐形眼镜的决策树
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    # 画图可视化展现
    dtPlot.createPlot(lensesTree)

if __name__ == '__main__':
    ContactLensesTest()
