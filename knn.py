#-*- encoding:utf-8 -*-
from numpy import *
from operator import itemgetter
#创建训练数据集
def createdata():
    data=array([[1.0,200.0,30.0],[0.9,300.0,20.0],[0.8,600.0,20.0],[0.7,400.0,48.0]])
    label = array(['A','A','B','B'])
    return data,label
def normalize_dataset(dataset):
    mindata = dataset.min(0)
    maxdata=dataset.max(0)
    rangedata = maxdata - mindata
    dataset = (dataset - mindata)/rangedata
    return dataset,mindata,rangedata
def normalize_data(data,mindata, rangedata):
    return (data - mindata)/rangedata
def classify(input_data, dataset,label,k):
    datalen =dataset.shape[0]
    sub_data = tile(input_data,(datalen,1))
    diff = dataset - sub_data
    sqdiff = diff**2
    dist = sqdiff.sum(axis=1) #各种特征的平方和
    dist = dist**0.5  #计算各种特征的距离
    #获取前K个值,距离最小的
    dist = dist.argsort()
    datadist = dist[:k]
    classcount = {}
    for index in datadist:
        classcount[label[index]]  =classcount.get(label[index],0) +1
    #对结果进行排序
    sortdata = sorted(classcount.iteritems(), key=itemgetter(1),reverse=True)
    return sortdata[0][0]
if __name__ == '__main__':
    testdata = [0.5,200.0,50.0]
    data, label = createdata()
    dataset, mindata, rangedata = normalize_dataset(data)
    test_data_res = normalize_data(testdata,mindata, rangedata)
    print(classify(test_data_res, dataset,label,12))


