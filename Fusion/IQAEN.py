from VGGnet.vgg16 import Vgg16
from imageio import imread
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt


from deepIQA_evaluate import IQA





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

	
num_1 = 32
num_2 = 33
for idx in range(num_1, num_2):
	img1 = imread('Test_vi_low/%d.bmp' % idx)
	img2 = imread('Test_ir_low/%d.bmp' % idx)
	img3 = imread('Test_vi_low/%d.bmp' % idx)
	img4 = imread('Test_ir_low/%d.bmp' % idx)
	img5 = imread('Test_vi_high/%d.bmp' % idx)
	Shape1=img1.shape
	h1 = Shape1[0]
	w1 = Shape1[1]
	Shape2 = img2.shape
	h2 = Shape2[0]
	w2 = Shape2[1]
	assert (h1 == h2 and w1 == w2), 'Two images must have the same shape!'
	print('input shape:', img1.shape)
	img1 = img1.reshape([1, h1, w1, 1])
	img2 = img2.reshape([1, h1, w1, 1])





def W(inputs1,inputs2, trained_model_path, w_en, c):
	# with tf.device('/gpu:1'):
	iqa1 = IQA(inputs = inputs1, trained_model_path = trained_model_path)
	iqa2 = IQA(inputs = inputs2, trained_model_path = trained_model_path)
	with tf.device('/cpu:0'):
		en1 = EN(inputs1)
		en2 = EN(inputs2)
		score1 = w_en *iqa1 +  en1
		score2 = w_en *iqa2 +  en2
		w1 = np.exp(score1 / c) / (np.exp(score1 / c) + np.exp(score2 / c))
		w2 = np.exp(score2 / c) / (np.exp(score1 / c) + np.exp(score2 / c))
		print('IQA_1: %f, IQA_2: %f' % (iqa1[0], iqa2[0]))
		print('EN_1: %f, EN_2: %f' % (en1[0], en2[0]))
		print('score_1: %f, score_2: %f' % (score1, score2))
		print('w1: %s, w2: %s\n' % (w1[0], w2[0]))
	return (w1,w2)

IQA_model = './IQA/models/nr_tid_weighted.model'
w_en=2
c=30


valid_w1, valid_w2 = W(inputs1 = img1, inputs2 = img2, trained_model_path = IQA_model,w_en=w_en,c=c)
print('valid_w1: %s, valid_w2: %s\n' % (valid_w1, valid_w2))



fig = plt.figure()
fig1 = fig.add_subplot(221)
fig2 = fig.add_subplot(222)
fig3 = fig.add_subplot(223)
fig4 = fig.add_subplot(224)
fig1.imshow(img1[0, :, :, 0], cmap = 'gray')
fig2.imshow(img2[0, :, :, 0], cmap = 'gray')

fused_low=valid_w1*img3+valid_w2*img4
fig3.imshow(fused_low,cmap='gray')
fused=fused_low+img5
fig4.imshow(fused,cmap='gray')
plt.show()