#!/usr/bin/env python
# coding=utf-8
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import corner_harris, corner_peaks
from scipy.ndimage import filters

def harris_eps(im,sigma = 3):
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im,(sigma,sigma),(0,1),imx)
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im,(sigma,sigma),(1.0),imy)
    #计算两两之间的一阶导数
    Wxx = filters.gaussian_filter(imx*imx,sigma)
    Wxy = filters.gaussian_filter(imx*imy,sigma)
    Wyy = filters.gaussian_filter(imy*imy,sigma)
    #计算行列式
    Wdet = Wxx * Wyy -Wxy**2
    #计算矩阵的迹
    Wtr = Wxx + Wyy
    #按eps公式计算
    return Wdet * 2 / (Wtr + 1e-06)

im1 = np.array(Image.open('pic.jpg').convert('L'))
im2 = np.array(Image.open('pic2.jpg').convert('L'))

my_coords1 = corner_peaks(harris_eps(im1,sigma = 1), min_distance = 12, threshold_rel = 0)
eps_coords1 = corner_peaks(corner_harris(im1,method = 'eps',sigma = 1),min_distance = 20,threshold_rel =0)
k_coords1 = corner_peaks(corner_harris(im1, method='k', sigma=1), min_distance=20, threshold_rel=0)

my_coords2 = corner_peaks(harris_eps(im2, sigma=1), min_distance=5, threshold_rel=0.01)
eps_coords2 = corner_peaks(corner_harris(im2, method='eps', sigma=5), min_distance=2, threshold_rel=0.01)
k_coords2 = corner_peaks(corner_harris(im2, method='k', sigma=1), min_distance=5, threshold_rel=0.01)

def plot_coords(index, title, im, coords):
    plt.subplot(index)
    plt.imshow(im)
    plt.plot(coords[:, 1], coords[:, 0], '+r', markersize=10)
    plt.title(title)
    plt.axis('off')

plt.gray()
#index = 321
index = 111
#plot_coords(index, 'my', im1, my_coords1)
#plot_coords(index+1 , 'my', im2, my_coords2)
#plot_coords(index + 2, 'skimage-eps', im1, eps_coords1)
plot_coords(index , 'skimage-eps', im2, eps_coords2)
#plot_coords(index + 4, 'skimage-k', im1, k_coords1)
#plot_coords(index + 5, 'skimage-k', im2, k_coords2)
plt.tight_layout(w_pad=0)
plt.show()


