# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:10:02 2020

@author: MonishRamadoss
"""
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, models, layers
from medpy.io import load
	
mask_list = []
image_list = []
for file in glob.glob('**/*.mhd',  recursive=True):
	if 'r.mhd' in file:
		mask_list.append(file)
	elif 'cti.mhd' in file:
		image_list.append(file)

def im_gen():
	data_set = list()
	label_set = list()
	for path0, path1 in zip(mask_list, image_list):
		image, header_img = load(path1)
		mask_img, header_mask = load(path0)
		mask_img = np.transpose(image, (2, 0, 1))
		image = np.transpose(image, (2,0,1))
		for i in range(image.shape[0]):
			data_set.append(np.expand_dims(image[i], 0))
			label_set.append(np.expand_dims(mask_img[i], 0))
	return np.array(data_set), np.array(label_set)

	
def normalize(x):
	x = np.asarray(x)
	return (x - x.min()) / (np.ptp(x))
	
if __name__ == '__main__':
	
	inputs = Input(shape=(1, 512, 512))
	
	# --- Define model
	# --- Define kwargs dictionary
	kwargs = {
		'kernel_size': 3,
		'padding': 'same'}
	
	# --- Define lambda functions
	conv = lambda x, filters, strides : layers.Conv2D(filters=filters, strides=strides, **kwargs)(x)
	norm = lambda x : layers.BatchNormalization()(x)
	relu = lambda x : layers.LeakyReLU()(x)
	
	# --- Define stride-1, stride-2 blocks
	conv1 = lambda filters, x : relu(norm(conv(x, filters, strides=1)))
	conv2 = lambda filters, x : relu(norm(conv(x, filters, strides=2)))
	# --- Define single transpose
	tran = lambda x, filters, strides : layers.Conv2DTranspose(filters=filters, strides=strides, **kwargs)(x)
	# --- Define transpose block
	tran2 = lambda filters, x : relu(norm(tran(x, filters, strides=2)))
	concat = lambda a, b : layers.Concatenate(axis=1)([a, b])
	
	
	l1 = conv1(16, inputs)
	l2 = conv1(32, conv2(16, l1))
	l3 = conv1(48, conv2(32, l2))
	l4 = conv1(64, conv2(48, l3))
	l5 = conv1(128, conv2(64, l4))
	l6 = tran2(64, l5)
	l7 = tran2(48, conv1(64, concat(l4, l6)))
	l8 = tran2(32, conv1(48, concat(l3, l7)))
	l9 = tran2(16, conv1(32, concat(l2, l8)))
	l10 = conv1(16, l9)
	logits = layers.Conv2D(filters=1, **kwargs)(l10)
	model = Model(inputs=inputs, outputs=logits)
	
	train_examples, train_labels = im_gen()
	train_labels = normalize(train_labels)
	tf_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
	tf_dataset = tf_dataset.shuffle(100).batch(1)
	print(train_examples.shape)

	optimizer=tf.keras.optimizers.Adam()
	loss= tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size')
	metrics = ['accuracy']
	model.compile(optimizer = optimizer, loss= loss, metrics= metrics)
	model.fit(train_examples, train_labels, epochs = 10)