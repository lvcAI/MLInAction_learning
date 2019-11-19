#!/usr/bin/env python
# -*- coding: UTF-8 -*-


"""
Created on Sep 16, 2010
Update  on 2017-05-18
Author: Peter Harrington/羊三/小瑶
GitHub: https://github.com/apachecn/AiLearning
"""

# K-邻近算法的一般流程
# 1.收集数据：可以使用任何方法
# 2.准备数据：距离计算所需要的数值，最好是结构化的数据结构格式
# 3.分析数据：可以使用任何方法
# 4.NativeBayes.训练算法：此步骤不适用与k-邻近算法
# 5.测试算法：计算错误率
# 6.使用算法：首先需要输入样本数据和结构化的输出结果，然后运行k-近邻算法判定输入数据分别属于哪个分类，最后应用对计算出的分类执行后续处理。

#

from numpy import *
import operator
import os
from collections import Counter


def createDataSet():
    """
       Desc:
           创建数据集和标签
       Args:
           None
       Returns:
           group -- 训练数据集的 features
           labels -- 训练数据集的 labels
       调用方式
       import kNN
       group, labels = kNN.createDataSet()
       """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 程序清单 2-1 K-邻近算法

def classify0(inX, dataSet, labels, k):
    """
       Desc:
           kNN 的分类函数
       Args:
           inX -- 用于分类的输入向量/测试数据
           dataSet -- 训练数据集的 features
           labels -- 训练数据集的 labels
           k -- 选择最近邻的数目
       Returns:
           sortedClassCount[0][0] -- 输入向量的预测分类 labels
       注意：labels元素数目和dataSet行数相同；程序使用欧式距离公式.
       预测数据所在分类可在输入下列命令
       kNN.classify0([0,0], group, labels, 3)
       """

    # 65-71 行为计算距离。采用两点之间的距离公式，欧氏距离
    # dataSetSize 等于 数据集 行数
    # 关于 shape  获取 数组的 行 列数
    #       使用方式:
    #                   dataSet.shape           out：（4.NativeBayes,2）
    #                   dataSet.shape[0]        out:    4.NativeBayes
    #                   dataSet.shape[1]        out:    2
    dataSetSize = dataSet.shape[0]
    #print(dataSetSize)
    #print(tile(inX, (dataSetSize, 1)))
    # tile  复制行列数（扩充）
    # tile(inX,(2,3)) 表示对 行  复制 inX 2 倍，在 列 上 复制inX 3倍
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet

    #print("diffMat " + str(diffMat))
    sqDiffmat = diffMat ** 2
    #print(sqDiffmat)
    # sum  矩阵求和
    #       使用方式:
    #               sum()           矩阵中所有的点都加起来求和
    #               sum(axis=1)     分别对矩阵中的行中的点求和
    #               sum(axis=0)     分别对矩阵中的列中的点求和
    sqDistances = sqDiffmat.sum(axis=1)
    #print(sqDistances)
    distances = sqDistances ** 0.5
    #print(distances)
    # argsort
    # 参考链接  https://blog.csdn.net/iboxty/article/details/44975575
    sortedDistIndicies = distances.argsort()
    #print(sortedDistIndicies)
    classCount = {}

    # ------------------- 实现失败  --------------------------------
    # 参考 第一章 机器学习的数据基础 《机器学习算法原理与编程实践》  的 python 实现欧式距离
    # # 欧氏距离： 点到点之间的距离
    # osDistances = sqrt((tile(inX,(dataSetSize,1)) - dataSet)*(tile(inX,(dataSetSize,1))-dataSet).T)
    # sortedDistIndicies2 = distances.argsort()
    # 使用 numpy np.linalg.norm(vec1-vec2)
    osDistances = linalg.norm(inX - dataSet)
    sortedDistIndicies2 = distances.argsort()
    # -------------------------------------------------------------------------------------------------------------
    for i in range(k):
        #print("i=" + str(i))
        voteIlabel = labels[sortedDistIndicies2[i]]
        #print(sortedDistIndicies[i])
        #print(voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        #print(classCount[voteIlabel])
        #print(classCount)
    # 分类排序
    sortedClassCunt = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCunt[0][0]


# 测试classify0
def test1():
    groups, labels = createDataSet()
    print(str(groups))
    print(str(labels))
    print(classify0([0.5, 0.5], groups, labels, 3))


# ---------------------------------------------------------------------------------------------------------------------
# 示例 2：使用 K-近邻算法 改进约会网站的配对效果

# 开发流程
# 准备数据：从文本文件中解析数据
def file2matrix(filename):
    # 打开文件
    fr = open(filename, 'r')
    # 获取文件数据中的数据行的行数
    numberOfLines = len(fr.readlines())

    # 生成对应的空矩阵
    returnMat = zeros((numberOfLines, 3))  # 返回的 预处理 矩阵
    classLabelVector = []  # prepare labels return

    fr = open(filename, 'r')
    index = 0
    for line in fr.readlines():
        # str.strip([chars]) --返回移除字符串头尾指定的字符生成的新字符串
        line = line.strip()
       # print(line)
        # 以 '\t' 切割字符串
        listFromLine = line.split('\t')
       # print(listFromLine)
        # 每列的属性数据，即 features
        returnMat[index] = listFromLine[0:3]
        # 每列的类别数据，即 label 标签数据
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    # 返回数据矩阵 returnMat 和对应的类别 classLabelsVector
    return returnMat,classLabelVector

def autoNorm(dataSet):
    """
       Desc：
           归一化特征值，消除属性之间量级不同导致的影响
       Args：
           dataSet -- 需要进行归一化处理的数据集
       Returns：
           normDataSet -- 归一化处理后得到的数据集
           ranges -- 归一化处理的范围
           minVals -- 最小值

       归一化公式：
           Y = (X-Xmin)/(Xmax-Xmin)
           其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。

        知识点补充:数据归一化
            归一化是一种简化计算的方式，即将有量纲的表达式，经过变换，转换为为无量纲的表达式，称为标量。归一化是机器学习的一项基础工作。
            归一化方法有两种形式，一种是把数变为（0,1）之间的小数，一种是把有量纲表达式变为无量纲表达式。
    """
    # 计算每种熟悉那个的最大值、最小值、范围
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 极差
    ranges = maxVals - minVals
    # -------第一种实现方式---start-------------------------
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 生成与最小值之差组成的矩阵
    normDataSet= dataSet - tile(minVals,(m,1))
    # 将最小值之差初一范围组成矩阵
    normDataSet = dataSet / tile(ranges,(m,1))
    # -------第一种实现方式---end---------------------------------------------

    return normDataSet,ranges,minVals


def datingClassTest():
    """
       Desc：
           对约会网站的测试方法，并将分类错误的数量和分类错误率打印出来
       Args：
           None
       Returns：
           None
       """
    # 设置测试数据的的一个比例（训练数据集比例=1-hoRatio）
    hoRatio = 0.5

    datingDataMat, datingLabels = file2matrix("D:\CodingWork\MLwork\MLInAction_learning\src\db\\2.KNN\\datingTestSet2.txt")

    normMat,ranges, minVals = autoNorm(datingDataMat)
    # m 表示数据的行数，即矩阵的第一维
    m = normMat.shape[0]
    # 设置测试的样本数量， numTestVecs:m表示训练样本的数量
    numTestVecs = int(m * hoRatio)
    print('numTestVecs=', numTestVecs)
    errorCount = 0
    for i in range(numTestVecs):
        # 对数据测试
        classifierResult = classify0(normMat[i], normMat[numTestVecs : m], datingLabels[numTestVecs : m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        errorCount += classifierResult != datingLabels[i]
    print("the total error rate is: %f" % (errorCount / numTestVecs))
    print(errorCount)


# 测试 实例2
def test2():
    returnMat, classLabelVector =  file2matrix("D:\CodingWork\MLwork\MLInAction_learning\src\db\\2.KNN\\datingTestSet2.txt")
   # print(returnMat)
    # print(classLabelVector)

if __name__ == '__main__':
    datingClassTest()
