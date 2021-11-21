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
    parser.add_argument('--feaSec', default='index', type=str, help='feather select ways: index or var')
    parser.add_argument('--disMes', default='inf', type=str, help='distance messure: 1 or 2 or inf')
    parser.add_argument('--pca', action="store_true", default=False, help='PCA or not？')  #是否进行PCA降维
    parser.add_argument('--pcaDim', default=50, type=int, help='PCA n components')  #降到多少维
    args = parser.parse_args()

    #加载数据
    (train_data, train_labels), (test_data, test_labels) = load_minist('./data')
    print ("mnist data loaded!")


    if args.pca:    #使用PCA降维
        #降到多少维
        pca = PCA(n_components = args.pcaDim)
        pca.fit(train_data) #fit PCA with training data instead of the whole dataset
        train_data_pca = pca.transform(train_data)
        test_data_pca = pca.transform(test_data)

        t1=time.time()
        #对降维后数据进行建树
        kd_tree=KDtree(train_data_pca[:args.trn],train_labels[:args.trn],dis_mes=args.disMes,fea_sec=args.feaSec) #创建KD树
        t2=time.time()
        #测试数据
        count=0
        for sample,label in zip(test_data_pca[:args.ten],test_labels[:args.ten]):
            d,nearest_point=kd_tree.find_nearest(sample)
            print('最邻近点标签：',nearest_point.label)
            print('样本点标签：',label)
            if nearest_point.label==label:
                count+=1
        t3=time.time()

        if args.feaSec=='index':
            print('方法：index法')
        else:
            print('方法：var法（最大方差法）')
        print('%s train_samples 建树时间：%d s' %(args.trn,t2-t1))
        print('准确率：',count/args.ten)
        print('%s test_samples 找最近邻点消耗时间：%d s'%(args.ten,t3-t2))

    else:   #不使用PCA

        t1=time.time()
        #对降维后数据进行建树
        kd_tree=KDtree(train_data[:args.trn],train_labels[:args.trn],dis_mes=args.disMes,fea_sec=args.feaSec) #创建KD树
        t2=time.time()
        #测试数据
        count=0
        for sample,label in zip(test_data[:args.ten],test_labels[:args.ten]):
            d,nearest_point=kd_tree.find_nearest(sample)
            print('最邻近点标签：',nearest_point.label)
            print('样本点标签：',label)
            if nearest_point.label==label:
                count+=1
        t3=time.time()
        if args.feaSec=='index':
            print('方法：index法')
        else:
            print('方法：var法（最大方差法）')
        print('%s train_samples 建树时间：%d s' %(args.trn,t2-t1))
        print('准确率：',count/args.ten)
        print('%s test_samples 找最近邻点消耗时间：%d s'%(args.ten,t3-t2))


