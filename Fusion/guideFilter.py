import numpy as np
from imageio import imread
import math
# import torch
# import torch.nn
import cv2 
def guideFilter(I, p, winSize, eps, s):
    
    #输入图像的高、宽
    h, w = I.shape[:2]
    
    #缩小图像
    size = (int(round(w*s)), int(round(h*s)))
    
    small_I = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)
    small_p = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)
    
    #缩小滑动窗口
    X = winSize[0]
    small_winSize = (int(round(X*s)), int(round(X*s)))
    
    # I的均值平滑
    mean_small_I = cv2.blur(small_I, small_winSize)
    
    #p的均值平滑
    mean_small_p = cv2.blur(small_p, small_winSize)
    
    #I*I和I*p的均值平滑
    mean_small_II = cv2.blur(small_I*small_I, small_winSize)
    
    mean_small_Ip = cv2.blur(small_I*small_p, small_winSize)
    
    #方差
    var_small_I = mean_small_II - mean_small_I * mean_small_I #方差公式
    
    #协方差
    cov_small_Ip = mean_small_Ip - mean_small_I * mean_small_p
   
    small_a = cov_small_Ip / (var_small_I + eps)
    small_b = mean_small_p - small_a*mean_small_I
    
    #对a、b进行均值平滑
    mean_small_a = cv2.blur(small_a, small_winSize)
    mean_small_b = cv2.blur(small_b, small_winSize)
    
    #放大
    size1 = (w, h)
    mean_a = cv2.resize(mean_small_a, size1, interpolation=cv2.INTER_LINEAR)
    mean_b = cv2.resize(mean_small_b, size1, interpolation=cv2.INTER_LINEAR)
    
    q = mean_a*I + mean_b
    
    return q