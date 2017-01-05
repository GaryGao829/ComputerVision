#!/usr/bin/env python
# coding=utf-8
import numpy  as np
import os
import matplotlib.pyplot as plt
from PIL import Image

def getimpaths(datapath):
    paths = []
    for dir in os.listdir(datapath):
        try :
            for filename in os.listdir(os.path.join(datapath,dir)):
                paths.append(os.path.join(datapath,dir,filename))
        except:
            pass

    return paths

impaths = getimpaths('./att_faces')
m,n = np.array(Image.open(impaths[0])).shape[0:2] #图片的分辨率

X = np.mat([np.array(Image.open(impath)).flatten() for impath in impaths ]).T
print 'X.shape=',X.shape

def pca(X):
    dim,num_data = X.shape #dim : No. of dimension ; num_data : No. of sample
    mean_X = X.mean(axis = 1) #求的平均脸，axis = 1 表示计算每行的均值，结果为列向量
    X = X - mean_X #零均值化
    
    M = np.dot(X.T,X)
    e,EV = np.linalg.eigh(M)
    print 'e=',e.shape,e
    print 'EV=',EV.shape,EV

#    compat trick:
#    把原来要计算XX^(X'为X的转置) 的协方差矩阵（设为C）变换为计算X^X的协方差矩阵(设为C')
#    设E为C对应的特征向量矩阵，E'为C'对应的特征向量矩阵。则E＝XE'
#    最后再把E归一化

    tmp = np.dot(X,EV).T # 因为上面使用了compat trick 所以需要变换
    print 'tmp=',tmp.shape,tmp
    V = tmp[::-1]
    print 'V = ',V.shape,V

    for i in range(EV.shape[1]):
        V[:,i] /= np.linalg.norm ( EV[:,i]) #因为使用了compat trick 所以要进行归一化,这里np.linglg.norm() 方法求的是矩阵的范数，范数分为L0，L1，L2等好几种，这个方法默认求的是L2范数

    return V,EV,mean_X

V,EV,immean = pca(X)

#显示平均脸
plt.gray()
plt.subplot(2,4,1)
plt.imshow(immean.reshape(m,n))

#选前面7个特征脸 显示在后面的7个格子里
for i in range(7):
    plt.subplot(2,4,i+2)
    plt.imshow(V[i].reshape(m,n))

plt.show()

#求欧几里得距离
def euclidean_distance(p,q):
    p = np.array(p).flatten()
    q = np.array(q).flatten()
    return np.sqrt(np.sum(np.power((p-q),2)))


