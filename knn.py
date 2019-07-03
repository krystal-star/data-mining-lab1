# knn

import numpy as np


def DataSet():   #训练集
    labels = []
    trainingset = []
    testset = []
    traininglabels = []
    '''从文件中读取数据'''
    f = open('file_path', 'r')
    data = f.read()
    data = data.split('\n')
    for i in range(len(data)):
        data[i] = data[i].split(',')
        labels.append(data[i][len(data[i]) - 1]) #单独保存数据集的labels
        data[i].pop(len(data[i]) - 1)

        if i <= len(data)/3*2:    #将文件中前2/3数据保存为训练集，剩余1/3作为测试集
            trainingset.append(data[i])
            traininglabels.append(labels[i])
        else:
            testset.append(data[i])
    f.close()

    return trainingset, traininglabels, testset


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
        print('The predicted label for ', testset[i], ' is ', max(storeLabels[i], key=storeLabels[i].count))  #选取storeLabels中出现次数最多的labels



if __name__=='__main__':
    k = 10
    trainingset, labels, testset = DataSet()
    knn(testset, trainingset, labels, k)


