from VGGnet.vgg16 import Vgg16
from imageio import imread
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from deepIQA_evaluate import IQA
import os
import cv2
import math
from scipy import signal
fused_path = 'fused_low' #B1
fused_result='fused_test' #B1+B2+B3
fused_path3='fused_2' #B1+B2

def EN(inputs):
	len = inputs.shape[0]
	entropies = np.zeros(shape = (len, 1))
	grey_level = 256
	patch_size = 64
	counter = np.zeros(shape = (grey_level, 1))

	for i in range(len):
		input_uint8 = (inputs[i, :, :, 0] * 255).astype(np.uint8)
		input_uint8 = input_uint8 + 1
		for m in range(patch_size):
			for n in range(patch_size):
				indexx = input_uint8[m, n]
				counter[indexx] = counter[indexx] + 1
		total = np.sum(counter)
		p = counter / total
		for k in range(grey_level):
			if p[k] != 0:
				entropies[i] = entropies[i] - p[k] * np.log2(p[k])
	return entropies


def W(inputs1,inputs2, trained_model_path, w_en, c1,c2):
	

	iqa1 = IQA(inputs = inputs1, trained_model_path = trained_model_path)
	iqa2 = IQA(inputs = inputs2, trained_model_path = trained_model_path)
	with tf.device('/cpu:0'):
		en1 = EN(inputs1)
		en2 = EN(inputs2)
		score1 = iqa1 +  w_en *en1
		score2 = iqa2 + w_en * en2
		w1 = np.exp(score1 / c1) / (np.exp(score1 / c1) + np.exp(score2 / c1))
		w2 = np.exp(score2 / c2) / (np.exp(score1 / c2) + np.exp(score2 / c2))
        
	
	return (w1,w2)


num_1 = 1
num_2 =2
for idx in range(num_1, num_2):
    vi_low = imread('Test_vi_low/%d.bmp' % idx)
    ir_low = imread('Test_ir_low/%d.bmp' % idx)
    fused_high = imread('FusionNet_Results/%d.bmp' % idx)
    ir_low2 = imread('Test_ir_low2/%d.bmp' % idx)
    vi_low2 = imread('Test_vi_low2/%d.bmp' % idx)
    vi_o = imread('Test_vi_original/%d.bmp' % idx)
    ir_o= imread('Test_ir_original/%d.bmp' % idx)

    
   
    Shape1=vi_low.shape
    h1 = Shape1[0]
    w1 = Shape1[1]
    Shape2 = ir_low.shape
    h2 = Shape2[0]
    w2 = Shape2[1]
    assert(h1 == h2 and w1 == w2),'Two images must have the same shape!'

    img1 = vi_low.reshape([1, h1, w1, 1]) 
    img2 = ir_low.reshape([1, h1, w1, 1])
    IQA_model='./IQA/models/nr_tid_weighted.model'
  
    w_en=13  

    c1=50
    c2=20  
    valid_w1, valid_w2 = W(inputs1 = img1, inputs2 = img2, trained_model_path = IQA_model,w_en=w_en,c1=c1,c2=c2)
    print('valid_w1: %s, valid_w2: %s\n' % (valid_w1, valid_w2))

    start = time.time()
    fused_low=valid_w1*vi_low+valid_w2*ir_low 
    fused_low = cv2.normalize(fused_low, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) 
    image_path = os.path.join(os.getcwd(), fused_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    num = "%d" % (idx)
    image_path = os.path.join(image_path, num + ".bmp")
    cv2.imwrite(image_path,fused_low)

    fused2 = np.zeros((h1,w1))
    for i in range(h1): 
       for j in range(w1):
           fused2[i,j] = ir_low2[i,j] if vi_low2[i,j]<ir_low2[i,j] else vi_low2[i,j]      
    
    image_path3 = os.path.join(os.getcwd(), fused_path3)
    if not os.path.exists(image_path3):
       os.makedirs(image_path3)
    num = "%d" % (idx)
    image_path3 = os.path.join(image_path3, num + ".bmp")
    cv2.imwrite(image_path3,fused2) #B2

    fused2=fused_low+fused2 
    fused=0.5*fused2+0.5*fused_high  
    end = time.time()
    fused = cv2.normalize(fused, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) 
    image_path = os.path.join(os.getcwd(), fused_result)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    num = "%d" % (idx)
    image_path = os.path.join(image_path, num + ".bmp")
    cv2.imwrite(image_path,fused)
    print("Testing [%d] successfully,Testing time is [%f]" % (idx, end - start))