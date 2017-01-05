#!/usr/bin/env python
# coding=utf-8

# Programming computer vision with python 笔记第二篇
# https://segmentfault.com/a/1190000003946953
# 直方图均衡化(histogram equalization)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def histeq(im,nbr_bins = 256):
    imhist,bins = np.histogram(im.flatten(),nbr_bins,density = True ) #对每个元素求概率密度
    cdf = imhist.cumsum() #对概率密度求累积和
    cdf = 255 * cdf / cdf[-1] #累积和变换到0-255区间
    im2 = np.interp(im.flatten(),bins[:-1],cdf) #线性插值
    return im2.reshape(im.shape),cdf #还原图像维度

im = np.array(Image.open('pic2.jpg').convert('L'))
im2,cdf = histeq(im)

plt.gray()
plt.subplot(221)
plt.imshow(im)
plt.subplot(222)
plt.hist([x for x in im.flatten() if x < 250], 128)
plt.subplot(223)
plt.imshow(im2)
plt.subplot(224)
plt.hist([x for x in im2.flatten() if x < 250], 128)
plt.show()
