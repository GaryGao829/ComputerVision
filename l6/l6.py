#!/usr/bin/env python
# coding=utf-8


#图像处理 边缘检测常用算子
# http://blog.csdn.net/augusdi/article/details/9028331 

from PIL import Image
import numpy as np
from scipy.ndimage import filters
import matplotlib.pyplot as plt

im = np.array(Image.open('pic1.jpg').convert('L'))

#prewitt
pwimx = np.zeros(im.shape)
filters.prewitt(im,1,pwimx)
pwimy = np.zeros(im.shape)
filters.prewitt(im,0,pwimy)
pwmagnitude = np.sqrt(pwimx ** 2 + pwimy ** 2) # 计算两个向量的模，同np.hypot(pwimx,pwimy)

#sobel
sbimx = np.zeros(im.shape)
filters.sobel(im,1,sbimx)
sbimy = np.zeros(im.shape)
filters.sobel(im,0,sbimy)
sbmagnitude = np.sqrt(sbimx ** 2 + sbimy ** 2)

#gaussian
gsimx = np.zeros(im.shape)
filters.gaussian_filter(input = im,sigma =1 , order = (0,1),output = gsimx )
gsimy = np.zeros(im.shape)
filters.gaussian_filter(input = im,sigma =1 , order = (1,0),output = gsimy )
gsmagnitude = np.sqrt(gsimx ** 2 + gsimy ** 2)

plt.gray()

index = 221
plt.subplot(index)
plt.imshow(im)
plt.title('original')
plt.axis('off')

plt.subplot(index + 1)
plt.imshow(pwmagnitude)
plt.title('prewitt')
plt.axis('off')

plt.subplot(index + 2)
plt.imshow(sbmagnitude)
plt.title('prewitt')
plt.axis('off')

plt.subplot(index + 3)
plt.imshow(gsmagnitude)
plt.title('gaussian')
plt.axis('off')

plt.show()
