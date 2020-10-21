import os
import glob
import shutil
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow import optimizers, losses
from tensorflow.keras import Input, Model, layers, backend
import datetime
from tqdm import trange


from scipy.ndimage.filters import gaussian_filter
from scipy import signal
from scipy.ndimage import zoom
from scipy import ndimage
from scipy import interpolate
import scipy.ndimage

#import tensorflow_addons as tfa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from jarvis.train import custom, params
from jarvis.utils.general import gpus
gpus.autoselect(1)

p = params.load(csv='./hyper.csv', row=0)
os.makedirs(p['output_dir'], exist_ok=True)
MODEL_NAME = '{}/ckp/model.hdf5'.format(p['output_dir'])
log_dir = "{}/logs/".format(p['output_dir']) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def save_array(path, array, name):
    fig = plt.figure()
    ims = [[plt.imshow(array[i, 0, ...], animated=True)] for i in range(array.shape[0])]
    #im[0].save(path + name, save_all=True, append_images=im[1:])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(os.path.join(path, name))
    
if(not os.path.isdir(log_dir)):
    os.makedirs(log_dir)
    
shutil.rmtree('./image')
os.makedirs('./image')
    
# Data
def plaque_transform(cls, model=None):
    t = np.linspace(-10, 10, 30)
    bump = np.exp(-0.1*t**2)
    kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
    kernel = np.expand_dims(kernel, (0, -1))
    struct1 = ndimage.generate_binary_structure( 4, 1)
    dilate = lambda a : ndimage.binary_dilation(a, structure=struct1, iterations=5)
    
    def transform(data, label):
        data = data[1:-1]
        label = label[1:-1] 
        idxs = []
        for i in range(data.shape[0]):
            if np.count_nonzero(label[i] == cls):
                idxs.append(i)
        if len(idxs) > 2:
            data, label = data[idxs[0]:idxs[-1], ...], label[idxs[0]:idxs[-1], ...], 
            data = np.clip(data, -1024, 400) / 200
            data, label = np.expand_dims(data, (1)), np.expand_dims(label, (1))
            return data, label
        return None, None
    return transform

def thoracic_transform():
    crop = 512
    zoom = 3
    centerx = 60
    centery = 45
    def transform(data, label):
        min = np.min(data)
        d, l = [], []
        data = ndimage.zoom(data, (1, zoom, zoom), order=0)
        label = ndimage.zoom(label, (1, zoom, zoom), order=0)
        
        shape = data.shape[-2]
        start = shape//2 - crop//2
        data = data[:, start-centerx:start+crop-centerx, start-centery:start+crop-centery]
        label = label[:, start-centerx:start+crop-centerx, start-centery:start+crop-centery]
        lx, ly = data.shape[-2], data.shape[-1]  
        X, Y = np.ogrid[0:lx, 0:ly]
        mask = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > lx * ly / 4
        data[:,mask] = min
        label[:,mask] = 0
        idxs = []
        for i in range(data.shape[0]):
            if np.count_nonzero(label[i] != 0):
                idxs.append(i)
        if len(idxs) > 2:
            data, label = data[idxs[0]:idxs[-1], ...], label[idxs[0]:idxs[-1], ...], 
            data = np.clip(data, -1024, 400) / 200  
            data, label = np.expand_dims(data, (1)), np.expand_dims(label, (1))
            return data, label
        return None, None
    return transform


def data_process(data_path, batch_size=4, transform=None, train_percent=0.9, save_images=True):
    data_array = None
    label_array = None
    count = 0
    data_paths = os.listdir(data_path)
    for f in range(len(data_paths)):#, desc='processing data files'):
        file = data_paths[f]
        main_path = os.path.join(data_path, file)
        d_path = glob.glob(os.path.join(main_path, "*cti.npy"))[0]
        l_path = glob.glob(os.path.join(main_path, "*r.npy"))[0]
        data, label = transform(np.load(d_path), np.load(l_path))
        if data is not None:
            data_array = data if data_array is None else np.concatenate([data_array, data])               
            label_array = label if label_array is None else np.concatenate([label_array, label])
            if save_images:
                
                if not os.path.isdir('./image/{}'.format(data_path.split('/')[-1])):
                    os.makedirs('./image/{}'.format(data_path.split('/')[-1]))
                
                save_array('./image/{}'.format(data_path.split('/')[-1]), data, file+'_data_.gif')
                save_array('./image/{}'.format(data_path.split('/')[-1]), label, file+'_label_.gif')
    
    data_array = np.expand_dims(data_array, (1, -1)).astype(np.float32)
    label_array = np.expand_dims(label_array, (1, -1))
    assert(data_array.shape == label_array.shape)
    sz = data_array.shape[0]
    train_dataset = tf.data.Dataset.from_tensor_slices((data_array[:int(sz*train_percent)], label_array[:int(sz*train_percent)])).shuffle(100).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((data_array[int(sz*train_percent):], label_array[int(sz*train_percent):])).batch(1)
    return train_dataset, test_dataset


# MODEL
def dense_unet(inputs, filters=32, fs=1):
    '''Model Creation'''
    # Define kwargs dictionary
    kwargs = {
        'kernel_size': (1,3,3),
        'padding': 'same',
        'bias_initializer':'zeros'
    } 
    # Define lambda functions#
    conv = lambda x, filters, strides : layers.Conv3D(filters=int(filters), strides=(1, strides, strides), **kwargs)(x)
    norm = lambda x : layers.BatchNormalization()(x)
    relu = lambda x : layers.LeakyReLU()(x)
    # Define stride-1, stride-2 blocks#
    conv1 = lambda filters, x : relu(norm(conv(x, filters, strides=1)))
    conv2 = lambda filters, x : relu(norm(conv(x, filters, strides=2)))
    # Define single transpose#
    tran = lambda x, filters, strides : layers.Conv3DTranspose(filters=int(filters), strides=(1,strides, strides), **kwargs)(x)
    # Define transpose block#
    tran2 = lambda filters, x : relu(norm(tran(x, filters, strides=2)))
    concat = lambda a, b : layers.Concatenate()([a, b])
    # Define Dense Block#
    
    def dense_block(filters,input,DB_depth):
        ext = 2+DB_depth
        outside_layer = input
        for _ in range(0,int(ext)):
            inside_layer= conv1(filters, outside_layer)
            outside_layer = concat(outside_layer,inside_layer)
        return outside_layer
    
    def td_block(conv1_filters,conv2_filters,input,DB_depth):
        TD = conv1(conv1_filters,conv2(conv2_filters,input))
        DB = dense_block(conv1_filters,TD, DB_depth)
        return DB
    def tu_block(conv1_filters,tran2_filters,input,td_input,DB_depth):
        TU = conv1(conv1_filters,tran2(tran2_filters,input))
        C = concat(TU,td_input)
        DB = dense_block(conv1_filters,C, DB_depth)
        return DB
    
    # Build Model#
    # TD = convolutions that train down, DB = Dense blocks, TU = Transpose convolutions that train up, C = concatenation groups.
    
    TD1 = td_block(filters*1,filters*1, inputs['dat'],0*fs)
    TD2 = td_block(filters*1.5,filters*1,TD1,1*fs)
    TD3 = td_block(filters*2,filters*1.5,TD2,2*fs)
    TD4 = td_block(filters*2.5,filters*2,TD3,3*fs)
    TD5 = td_block(filters*3,filters*2.5,TD4,4*fs)
    
    TU1 = tu_block(filters*2.5,filters*3,TD5,TD4,4*fs)
    TU2 = tu_block(filters*2,filters*2.5,TU1,TD3,3*fs)
    TU3 = tu_block(filters*1.5,filters*2,TU2,TD2,2*fs)
    TU4 = tu_block(filters*1,filters*1.5,TU3,TD1,1*fs)
    TU5 = tran2(filters*1, TU4) 
    logits = {}
    logits['lbl'] = layers.Conv3D(filters = 2, name='lbl', **kwargs)(TU5)
    model = Model(inputs=inputs, outputs=logits )
    return model


# LOSS



def dsc_soft(weights=None, scale=1.0, epsilon=0.01, cls=1):
    @tf.function
    def dsc(y_true, y_pred):
        true = tf.cast(y_true[..., 0] == cls, tf.float32)
        pred = tf.nn.softmax(y_pred, axis=-1)[..., cls]
        if weights is not None:
            true = true * (weights[...]) 
            pred = pred * (weights[...])
        A = tf.math.reduce_sum(true * pred) * 2
        B = tf.math.reduce_sum(true) + tf.math.reduce_sum(pred) + epsilon
        return  (A / B) * scale
    return dsc


def sce(weights=None, scale=1.0):
    loss = losses.SparseCategoricalCrossentropy(from_logits=True)
    @tf.function
    def sce(y_true, y_pred):
        return loss(y_true=y_true, y_pred=y_pred, sample_weight=weights) * scale
    return sce


def happy_meal(alpha = 5, beta = 1, weights=None, epsilon=0.01, cls=1):
    l2 = sce(None, alpha)
    l1 = dsc_soft(None, beta, epsilon, cls)
    @tf.function
    def calc_loss(y_true, y_pred):
        return l2(y_true, y_pred) - l1(y_true, y_pred)
    return calc_loss

# TRAIN LOOP

def train():
    thoracic_train, thoracic_test = data_process('./data/Thoracic_Data', transform=thoracic_transform(),)
    
    
    plaque_train, plaque_test  = data_process('./data/Plaque_Data', transform=plaque_transform(3))
    
    
if __name__ == "__main__":
    train()