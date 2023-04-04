import numpy as np
from imageio import imread
import imageio
import math
import torch
import torch.nn
from guideFilter import *
import matplotlib.pyplot as plt

import os
num_1 = 32
num_2 = 33
for idx in range(num_1, num_2):
    vi = imread('Train_vi_original/%d.bmp' % idx)
    ir = imread('Train_ir_original/%d.bmp' % idx)
    #将图像归一化
    vi_0_1 = vi/255.0
    ir_0_1 = ir/255.0



    irlow_path = 'Train_ir_low'
    irhigh_path = 'Train_ir_high'
    vilow_path = 'Train_vi_low'
    vihigh_path = 'Train_vi_high'
    

    train_irlow_path = os.path.join(os.getcwd(), irlow_path)
    if not os.path.exists(train_irlow_path):
        os.makedirs(train_irlow_path)
    num = "%d" % idx
    train_irlow_path = os.path.join(train_irlow_path, num + ".bmp")

    train_irhigh_path = os.path.join(os.getcwd(), irhigh_path)
    if not os.path.exists(train_irhigh_path):
        os.makedirs(train_irhigh_path)
    num = "%d" % idx
    train_irhigh_path = os.path.join(train_irhigh_path, num + ".bmp")

    train_vilow_path = os.path.join(os.getcwd(), vilow_path)
    if not os.path.exists(train_vilow_path):
        os.makedirs(train_vilow_path)
    num = "%d" % idx
    train_vilow_path = os.path.join(train_vilow_path, num + ".bmp")

    train_vihigh_path = os.path.join(os.getcwd(), vihigh_path)
    if not os.path.exists(train_vihigh_path):
        os.makedirs(train_vihigh_path)
    num = "%d" % idx
    train_vihigh_path = os.path.join(train_vihigh_path, num + ".bmp")
   


    gf_vi_low = guideFilter(vi_0_1, vi_0_1, (16,16), math.pow(0.1,2),0.5)
    gf_vi_low = gf_vi_low*255
    gf_vi_low[gf_vi_low>255] = 255
    gf_vi_low = np.round(gf_vi_low)
    gf_vi_low = gf_vi_low.astype(np.uint8)
    cv2.imwrite( train_vilow_path,gf_vi_low)

    gf_ir_low = guideFilter(ir_0_1, ir_0_1, (16,16), math.pow(0.1,2),0.5)
    gf_ir_low = gf_ir_low*255
    gf_ir_low[gf_ir_low>255] = 255
    gf_ir_low = np.round(gf_ir_low)
    gf_ir_low = gf_ir_low.astype(np.uint8)
    cv2.imwrite(train_irlow_path,gf_ir_low)

    print("this is vi")
    print(vi_0_1)
    print("this is vi_low")
    print(gf_vi_low/255)
    gf_vi_high=vi_0_1-gf_vi_low/255

    print("this is high")
    print(gf_vi_high)
   
    
    gf_vi_high = cv2.normalize(gf_vi_high, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite(train_vihigh_path,gf_vi_high)

    plt.imshow(gf_vi_high,cmap='Greys')
    plt.title('gf_vi_high')
    plt.axis('off')
    plt.show()

    gf_ir_high=ir_0_1-gf_ir_low/255
   
    gf_ir_high = cv2.normalize(gf_ir_high, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)   #将其换成uint8格式就能正常显示保存了
    cv2.imwrite(train_irhigh_path,gf_ir_high)


    plt.figure(figsize=(2,2)) #设置窗口大小
    plt.figure()
    plt.suptitle('Image') # 图片名称
    plt.subplot(221), plt.title('gf_vi_low')
    plt.imshow(gf_vi_low,cmap='Greys'), plt.axis('off')

    plt.subplot(222), plt.title('gf_vi_high')
    plt.imshow(gf_vi_high,cmap='Greys'), plt.axis('off') #这里显示灰度图要加cmap

    plt.subplot(223), plt.title('gf_ir_high')
    plt.imshow(gf_ir_low,cmap='Greys'), plt.axis('off')

    plt.subplot(224), plt.title('gf_ir_high')
    plt.imshow(gf_ir_high,cmap='Greys'), plt.axis('off')
    plt.show()

