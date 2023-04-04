import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os


mask_path='mask_path_train'
num_1 = 1
num_2 = 2

for idx in range(num_1, num_2):
    
    img1 = cv.imread('Train_ir_original/%d.bmp' % idx)
   
    mask_path = os.path.join(os.getcwd(), mask_path)
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
    num = "%d" % (idx)
    mask_path = os.path.join(mask_path, num + ".bmp")

    
    img1gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)

    ret, mask = cv.threshold(img1gray, 190, 255, cv.THRESH_BINARY)
  
    plt.title("mask")
    plt.xticks([]),plt.yticks()
    plt.show()
    cv.imwrite(mask_path,mask)