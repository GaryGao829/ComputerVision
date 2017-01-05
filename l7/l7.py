#!/usr/bin/env python
# coding=utf-8
from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt 

im = np.array(Image.open('count.png').convert('L'))
im = 1 * ( im < 128)  #把灰度图像转换为二值图，即灰度小于128的当成黑点，否则当作白点

label_from_origin , num_from_origin = ndimage.measurements.label(im)
im_open = ndimage.morphology.binary_opening(im,np.ones((9,9)),iterations = 2) #用一个9*5全1的结构元素，并连续作两次开运算
label_from_open, num_from_open = ndimage.measurements.label(im_open)

#用原图显示
index = 221 
plt.subplot(index)
plt.imshow(im)
plt.title('origin')
plt.axis('off')

plt.subplot(index+1)
plt.imshow(label_from_origin)
plt.title('%d objects' % num_from_origin)
plt.axis('off')

plt.subplot(index+2)
plt.imshow(im_open)
plt.title('apply open')
plt.axis('off')

plt.subplot(index+3)
plt.imshow(label_from_open)
plt.title('%d objects' % num_from_open)
plt.axis('off')

plt.show()
