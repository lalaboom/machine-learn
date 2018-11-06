#-*- encoding:utf-8 -*-
from math import log
import numpy as np
#计算当前系统的熵
def _calEntropy(dataSet):
    labelData = dataSet[:,-1]
    total = float(labelData.shape[0])
    entropy =0.0
    for i in np.unique(labelData):
        p = np.sum(labelData==i)/total
        entropy -= p*log(p,2)
    return entropy

#计算某个特征的熵
def _calFeatureEnt(dataSet, index):
    fv = dataSet[:,index]
    total = float(dataSet.shape[0])
    entropy = 0.0
    for i in np.unique(fv):
        subEnt = _calEntropy(split_dataset(dataSet,index, i))
        p = np.sum(fv==i)/total
        entropy += p*subEnt
    return entropy

#选择出最合适的特征
def choose_best_feature(dataSet):
    baseEnt = _calEntropy(dataSet)
    lenf = dataSet.shape[1] -1
    index = -1
    value =  0
    for i in range(lenf):
        ent = _calFeatureEnt(dataSet, i)
        if (baseEnt - ent) > value:
            value = baseEnt - ent
            index = i
    return index

#化分数据
def split_dataset(dataSet,feature_index, value):
    res = dataSet[:,feature_index] == value
    newdataSet = dataSet[res]
    newdataSet = np.delete(newdataSet,feature_index, axis=1)
    return newdataSet


#创建数据
def create_Tree(dataSet, label):
    if np.sum(dataSet[:,-1] == dataSet[-1][-1]) == dataSet.shape[0]:
        return dataSet[-1][-1]
    if dataSet.shape[1] == 1:
        return calMax(dataSet)
    best_index =  choose_best_feature(dataSet)
    getfeature = label[best_index]
    mytree = {getfeature:{}}
    del label[best_index]
    for item in np.unique(dataSet[:,best_index]):
        sublabel = label[:]                  #一定要用副本
        mytree[getfeature][item] = create_Tree(split_dataset(
            dataSet,best_index, item

        ),sublabel)
    return mytree

#如果最后特征用完了，还是不能划分出类别来，就只能选择出那个类最多的
#比如最后的数据是这样的 [['yes'],['no'],['yes']],这时候调calMax返回'yes'
def calMax(dataSet):
    dataSet = dataSet[:,-1]
    flag = 0
    res =dataSet[0]
    udata = np.unique(dataSet)
    for i  in udata:
        num = np.sum(dataSet==i)
        if num >= flag:
            flag = num
            res = i
    return res

if __name__ == '__main__':
    dataSet = [[1,1,'yes'],
           [1,1,'yes'],
           [1,0,'no'],
           [0,1,'no'],
           [0,1,'no']
    ]
    testdata = np.array(dataSet)
    labels = ['no surfacing', 'flippers']
    print(create_Tree(testdata,labels))