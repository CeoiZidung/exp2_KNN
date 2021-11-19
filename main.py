import numpy as np
import matplotlib.pyplot as plt
import time
from KDtree import *
from load_minist import *
from sklearn.decomposition import PCA
import argparse

if __name__=='__main__':
    #添加命令行指令
    parser = argparse.ArgumentParser(description='KNN_project')
    parser.add_argument('--trn', default=2000, type=int, help='train num')
    parser.add_argument('--ten', default=1000, type=int, help='test num')
    parser.add_argument('--feaSec', default='var', type=str, help='feather select ways: index or var')
    parser.add_argument('--disMes', default='2', type=str, help='distance messure: 1 or 2 or inf')
    parser.add_argument('--pcaDim', default=50, type=int, help='PCA n components')  #降到多少维
    args = parser.parse_args()

    #加载数据
    (train_images, train_labels), (test_images, test_labels) = load_minist('./data')
    print ("mnist data loaded")
    #将每张图片展开到一维
    train_data=train_images.reshape(60000,784)
    test_data=test_images.reshape(10000,784)
    print ("training data shape after reshape:",train_data.shape)
    print ("testing data shape after reshape:",test_data.shape)

    #降到多少维
    pca = PCA(n_components = args.pcaDim)
    pca.fit(train_data) #fit PCA with training data instead of the whole dataset
    train_data_pca = pca.transform(train_data)
    test_data_pca = pca.transform(test_data)
    print("PCA completed with 100 components")

    t1=time.time()
    #对降维后数据进行建树
    kd_tree=KDtree(train_data[:args.trn],train_labels[:int(args.trn)],dis_mes=args.disMes,fea_sec=args.feaSec) #创建KD树
    t2=time.time()

    #
    # for i in range(1,2000):
    #     node=Node(item=train_data_pca[i],label=train_labels[i])
    #     kd_tree.insert(node)

    #测试数据
    count=0
    for sample,label in zip(test_data[:args.ten],test_labels[:args.ten]):
        d,nearest_point=kd_tree.find_nearest(sample)
        print('最邻近点标签：',nearest_point.label)
        print('样本点标签：',label)
        if nearest_point.label==label:
            count+=1
    t3=time.time()


    print('%s train_samples 建树时间：%d s' %(args.trn,t2-t1))
    print('准确率：',count/args.ten)
    print('%s test_samples 找最近邻点消耗时间：%d s'%(args.ten,t3-t2))









