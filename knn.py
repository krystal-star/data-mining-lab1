# knn

import numpy as np
import csv

def DataSet():   #对于txt或data文件
    labels = []
    trainingset = []
    testset = []
    traininglabels = []
    '''从文件中读取数据'''
    f = open('fileName', 'r')
    data = f.read()
    data = data.split('\n')
    for i in range(len(data)):
        data[i] = data[i].split('\t')
        labels.append(data[i][len(data[i]) - 1]) #单独保存数据集的labels
        data[i].pop(len(data[i]) - 1)

        if i <= len(data)/3*2:    #将文件中前2/3数据保存为训练集，剩余1/3作为测试集
            trainingset.append(data[i])
            traininglabels.append(labels[i])
        else:
            testset.append(data[i])
    f.close()
    return trainingset, traininglabels, testset


def DataSetCSV():  #对于csv文件
    labels = []
    trainingset = []
    testset = []

    '''从文件中读取训练集'''
    f = open('/Users/liukai/Desktop/datamining/uci_data/ForestTypes/training.csv', 'r')
    data = csv.reader(f)
    for line in data:
        trainingset.append(line)
    trainingset.pop(0)
    for i in range(len(trainingset)):
        labels.append(trainingset[i][0])  # 单独保存数据集的labels
        trainingset[i].pop(0)
    f.close()
    '''从文件中读取测试集'''
    f = open('/Users/liukai/Desktop/datamining/uci_data/ForestTypes/testing.csv', 'r')
    data = csv.reader(f)
    for line in data:
        testset.append(line)
    testset.pop(0)
    for i in range(len(testset)):
        testset[i].pop(0)
    f.close()

    return trainingset, labels, testset


def ecludSim(a1, a2):   #计算欧式距离
    distance = 0
    for i in range(len(a1)):
        d1 = float(a1[i])
        d2 = float(a2[i])
        distance = distance + (np.square(d1-d2))
    return np.sqrt(distance)


def knn(testset, trainingset, labels, k):
    distance = [[] for i in range(len(testset))]
    sortedIndex = [[] for i in range(len(testset))]
    storeLabels = [[] for i in range(len(testset))]
    '''遍历测试集和训练集，计算欧式距离'''
    for i in range(len(testset)):
        for j in range(len(trainingset)):
            distance[i].append(ecludSim(testset[i], trainingset[j]))
        sortedIndex[i] = np.argsort(distance[i])  #返回distance里从小到大的index

    for i in range(len(testset)):
        for j in range(k):
            storeLabels[i].append(labels[sortedIndex[i][j]]) #按从小到大顺序将前k个labels放入storeLabels中
        print('The predicted class for ', testset[i], ' is ', max(storeLabels[i], key=storeLabels[i].count))  #选取storeLabels中出现次数最多的labels



if __name__=='__main__':
    trainingset, labels, testset = DataSetCSV()
    k = 7
    knn(testset, trainingset, labels, k)


