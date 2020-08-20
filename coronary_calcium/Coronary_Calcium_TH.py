import os
import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow import optimizers #, losses
# from tensorflow.keras import Input
import nibabel as nib

from PIL import Image
#import tensorflow_addons as tfa

from model import RA_UNET

def preprocess():
	ROOT = './data/Thoracic_Data'
	gz_files = list()
	tmp = []
	print(ROOT)
	for file in os.listdir(ROOT):
		print(file)
		data = 	nib.load(os.path.join(ROOT, file, 'data.nii.gz')).get_fdata()
		label = nib.load(os.path.join(ROOT, file, 'label.nii.gz')).get_fdata()
		for i in range(min(data.shape[-1], label.shape[-1])):
			data_path = os.path.join(ROOT, file, 'data_{0}.npy'.format(i))
			label_path = os.path.join(ROOT, file, 'label_{0}.npy'.format(i))

			# np.save(data_path, data[:,:,i])
			# np.save(label_path, label[:,:,i])
			
			d = data[:,:,i] * label[:,:,i]
			tmp.append([d, data[:,:,i]])
			gz_files.append([data_path, label_path])
			
		break
	id = 0
	for d, y in tmp:
		rescaled = (255.0 / d.max() * (d - d.min())).astype(np.uint8)
		im = Image.fromarray(rescaled)
		im.save("./images/data{0}_0.png".format(id))
		rescaled = (255.0 / y.max() * (y - y.min())).astype(np.uint8)
		im = Image.fromarray(rescaled)
		im.save("./images/data{0}_1.png".format(id))
		id += 1
	
	return gz_files

preprocess()