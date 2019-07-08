#k-means + PCA

import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition


def DataSet():   #对于txt或data文件
    '''从文件中读取数据'''
    f = open('fileName', 'r')
    data = f.read()
    data = data.split('\n')
    for i in range(len(data)):
        data[i] = data[i].split(',')
        data[i].pop(len(data[i]) - 1)
    f.close()

    '''利用SKlearn库自带PCA函数进行降维'''
    datamat = np.mat(data)
    dataset = decomposition.PCA(n_components=2).fit_transform(datamat)
    return dataset


def ecludSim(a1, a2):   #计算欧式距离
    distance = 0
    for i in range(2):
        d1 = a1[i]
        d2 = a2[i]
        distance += np.square(d1 - d2)
    return np.sqrt(distance)


def randomCenter(dataset, k):  #选取距离尽量远的k个点为centers
    centers = [[] for i in range(k)]
    d = []
    r = np.random.randint(0, len(dataset)+1)   #第一个中心点随机选取
    centers[0].append(dataset[r][0])
    centers[0].append(dataset[r][1])
    k -= 1
    flag = 0
    max = 0

    while k > 0:
        for i in range(len(dataset)):
            if ecludSim(dataset[i], centers[flag]) >= max:
                x = dataset[i][0]
                y = dataset[i][1]
                max = ecludSim(dataset[i], centers[flag])
        centers[flag+1].append(x)
        centers[flag+1].append(y)
        k -= 1
        flag += 1

    return centers


def kMeans(dataset, k):
    centers = randomCenter(dataset, k)
    change = True
    clusters = []

    while change:
        change = False

        d = [[] for i in range(len(dataset))]
        for i in range(len(dataset)):
            for j in range(k):
                d[i].append(ecludSim(dataset[i], centers[j]))  #计算每个例子到三个中心点的距离
            clusters.append(np.argmin(d[i])) #保存距离最小的中心点的下标

        for i in range(k):
            n = x = y = 0
            for j in range(len(dataset)):
                if clusters[j] == i:
                    x += float(dataset[j][0])
                    y += float(dataset[j][1])
                    n += 1

            if centers[i][0] != x/n or centers[i][1] != y/n:  #重新计算中心点坐标
                change = True
                centers[i][0] = x / n
                centers[i][1] = y / n
    return clusters


def visualization(dataset,clusters):
    color = ['#FF6347', '#FFD700', '#8A2BE2', '#66CDAA', '#8FBC8F']  #选了几个好看的颜色
    for i in range(len(dataset)):
        plt.scatter(float(dataset[i][0]), float(dataset[i][1]), c=color[int(clusters[i])])

    plt.show()


if __name__=='__main__':
    dataset = DataSet()
    k = 3
    clusters = kMeans(dataset, k)
    visualization(dataset, clusters)



