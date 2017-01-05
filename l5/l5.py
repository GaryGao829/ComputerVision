#!/usr/bin/env python
# coding=utf-8
from PIL import Image
from scipy.ndimage import filters
import numpy as np
import matplotlib.pyplot as plt

im = np.array(Image.open('/Users/yugao/Downloads/587887.jpg'))
index = 221
plt.subplot(index)
plt.imshow(im)

for sigma in (2,5,10):
    im_blur = np.zeros(im.shape,dtype = np.uint8)
    for i in range(3):
        im_blur[:,:,i] = filters.gaussian_filter(im[:,:,i],sigma)

    index += 1
    plt.subplot(index)
    plt.imshow(im_blur)

plt.show()
