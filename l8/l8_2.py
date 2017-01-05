#!/usr/bin/env python
# coding=utf-8

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from skimage.restoration import denoise_bilateral  #for new version

#im = np.array(Image.open('noising.jpg').convert('L'))
im = np.array(Image.open('noising.jpg'))

index = 221
plt.subplot(index)
plt.gray()
plt.imshow(im)
plt.axis('off')
plt.title("original")

plt.subplot(index+1)
plt.imshow(denoise_bilateral(im))
plt.axis('off')
plt.title("default")

plt.subplot(index+2)
plt.imshow(denoise_bilateral(im, sigma_range=0.2, sigma_spatial=10))
plt.axis('off')
plt.title("0.2/10")

plt.subplot(index+3)
plt.imshow(denoise_bilateral(im, sigma_range=0.8, sigma_spatial=10))
plt.axis('off')
plt.title("0.8/10")

plt.show()
