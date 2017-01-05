#!/usr/bin/env python
# coding=utf-8
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#pim = Image.open('pic.jpg').crop((120,370,460,675)).convert('1')
pim = Image.open('pic2.jpg')
pim.show()

#im = np.array(Image.open('pic.jpg').crop((120,370,460,675)).resize((256,230)).convert('L'))
im = np.array(Image.open('pic2.jpg').resize((256,230)).convert('L'))
n,m = im.shape[0:2]
points = []
for i in range(n):
    for j in range(m):
        if im[i,j] < 128.0: # 把小于128的灰度值当做黑点取出来
            points.append([float(j),float(n) - float(i)]) #坐标转换

im_X = np.mat(points).T #转置之后，行表示维度（x和y），列表示每个样本点
print 'im_X = ',im_X,'shape = ',im_X.shape

def pca(X, k=1): #降为k维
    d,n = X.shape
    mean_X = np.mean(X, axis=1) #axis为0表示计算每列的均值，为1表示计算每行均值
    print 'mean_X=',mean_X
    X = X - mean_X
    #计算不同维度间的协方差，而不是样本间的协方差，方法1：
    #C = np.cov(X, rowvar=1) #计算协方差，rowvar为0则X的行表示样本，列表示特征/维度
    #方法2：
    C = np.dot(X, X.T)
    e,EV = np.linalg.eig(np.mat(C)) #求协方差的特征值和特征向量
    print 'C=',C
    print 'e=',e
    print 'EV=',EV
    e_idx = np.argsort(-e)[:k] #获取前k个最大的特征值对应的下标（注：这里使用对负e排序的技巧，反而让原本最大的排在前面）
    EV_main = EV[:,e_idx]   #获取特征值（下标）对应的特征向量，作为主成分
    print 'e_idx=',e_idx,'EV_main=',EV_main      
    low_X = np.dot(EV_main.T, X)    #这就是我们要的原始数据集在主成分上的投影结果
    return low_X, EV_main, mean_X

low_X, EV_main, mean_X = pca(im_X)
print "low_X=",low_X
print "EV_main=",EV_main
recon_X = np.dot(EV_main, low_X) + mean_X  #把投影结果重构为二维表示，以便可以画出来直观的看到
print "recon_X.shape=",recon_X.shape

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(im_X[0].A[0], im_X[1].A[0],s=1,alpha=0.5)
ax.scatter(recon_X[0].A[0], recon_X[1].A[0],marker='o',s=100,c='blue',edgecolors='white')
plt.show()
