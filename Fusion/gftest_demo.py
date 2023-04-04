import numpy as np
from imageio import imread
import imageio
import math
import time
from guideFilter import *
import matplotlib.pyplot as plt

import os
num_1 = 1
num_2 = 2
for idx in range(num_1, num_2):
    vi = imread('Test_vi_original/%d.bmp' % idx)
    ir = imread('Test_ir_original/%d.bmp' % idx)

    vi_0_1 = vi/255.0
    ir_0_1 = ir/255.0


    irlow_path = 'Test_ir_low'
    irhigh_path = 'Test_ir_high'
    vilow_path = 'Test_vi_low'
    vihigh_path = 'Test_vi_high'
    fused_path = 'FusionNet_Results'

    fused_irlow_path = os.path.join(os.getcwd(), irlow_path)
    if not os.path.exists(fused_irlow_path):
        os.makedirs(fused_irlow_path)
    num = "%d" % idx
    fused_irlow_path = os.path.join(fused_irlow_path, num + ".bmp")

    fused_irhigh_path = os.path.join(os.getcwd(), irhigh_path)
    if not os.path.exists(fused_irhigh_path):
        os.makedirs(fused_irhigh_path)
    num = "%d" % idx
    fused_irhigh_path = os.path.join(fused_irhigh_path, num + ".bmp")

    fused_vilow_path = os.path.join(os.getcwd(), vilow_path)
    if not os.path.exists(fused_vilow_path):
        os.makedirs(fused_vilow_path)
    num = "%d" % idx
    fused_vilow_path = os.path.join(fused_vilow_path, num + ".bmp")

    fused_vihigh_path = os.path.join(os.getcwd(), vihigh_path)
    if not os.path.exists(fused_vihigh_path):
        os.makedirs(fused_vihigh_path)
    num = "%d" % idx
    fused_vihigh_path = os.path.join(fused_vihigh_path, num + ".bmp")
    
    fused_path = os.path.join(os.getcwd(), fused_path)
    if not os.path.exists(fused_path):
        os.makedirs(fused_path)
    num = "%d" % idx
    fused_path = os.path.join(fused_path, num + ".bmp")
   

    start = time.time()
    gf_vi_low = guideFilter(vi_0_1, vi_0_1, (40,40), math.pow(0.1,2),0.5)
    
    gf_vi_low = gf_vi_low*255
    gf_vi_low[gf_vi_low>255] = 255
    gf_vi_low = np.round(gf_vi_low)
    gf_vi_low = gf_vi_low.astype(np.uint8)
    cv2.imwrite( fused_vilow_path,gf_vi_low)

    gf_ir_low = guideFilter(ir_0_1, ir_0_1, (40,40), math.pow(0.1,2),0.5)
    gf_ir_low = gf_ir_low*255
    gf_ir_low[gf_ir_low>255] = 255
    gf_ir_low = np.round(gf_ir_low)
    gf_ir_low = gf_ir_low.astype(np.uint8)
    cv2.imwrite(fused_irlow_path,gf_ir_low)
    
    gf_vi_high=vi_0_1-gf_vi_low/255
    gf_vi_high = cv2.normalize(gf_vi_high, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite(fused_vihigh_path,gf_vi_high)

    gf_ir_high=ir_0_1-gf_ir_low/255
    gf_ir_high = cv2.normalize(gf_ir_high, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)   #将其换成uint8格式就能正常显示保存了
    cv2.imwrite(fused_irhigh_path,gf_ir_high)

    end = time.time()
    print("guideFilter [%d] successfully,Testing time is [%f]" % (idx, end - start))


