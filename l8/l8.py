#!/usr/bin/env python
# coding=utf-8
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters 
from skimage.restoration import denoise_tv_chambolle

im = np.array(Image.open('noising.jpg').convert('L'))

index = 221
plt.subplot(index)
plt.gray()
plt.imshow(im)
plt.axis('off')
plt.title("original")

chdnim = denoise_tv_chambolle(im,weight = 0.1)
plt.subplot(index +1)
plt.imshow(chdnim)
plt.axis('off')
plt.title("chambolle weight = 0.1")

gs2dnim = filters.gaussian_filter(im,sigma = 2)
plt.subplot(index + 2)
plt.imshow(gs2dnim)
plt.axis('off')
plt.title("gaussion sigma =2")

gs3dnim = filters.gaussian_filter(im,sigma =3 )
plt.subplot(index + 3)
plt.imshow(gs3dnim)
plt.title("gaussion sigmal =3")
plt.show()

