import os
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow import optimizers, losses
from tensorflow.keras import Input, Model, layers, backend
import datetime
#import tensorflow_addons as tfa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from jarvis.train import datasets, custom, params
from jarvis.train.client import Client
from jarvis.utils.general import overload, tools as jtools, gpus
gpus.autoselect(1)

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

paths = datasets.download(name='ct/structseg', path='./data')

p = params.load(csv='./hyper.csv', row=7)
configs = {'batch': {'size': p['batch_size'], 'fold': p['fold']}}
MODEL_NAME = '{}/ckp/model.hdf5'.format(p['output_dir'])
path = '{}/data/ymls/client-heart.yml'.format(paths['code'])
client = Client(path, configs=configs)
gen_train, gen_valid = client.create_generators()
inputs = client.get_inputs(Input)

log_dir = "{}/logs/".format(p['output_dir']) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if(not os.path.isdir(log_dir)):
    os.makedirs(log_dir)


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

@overload(Client)
def preprocess(self, arrays, **kwargs):   
    arrays['ys']['lbl'] = arrays['ys']['lbl'] 
    #msk = cv2.GaussianBlur(arrays['ys']['lbl'], (99,99), 0 )
    return arrays

def train():
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='{}/ckp/'.format(p['output_dir']),
        save_weights_only=True,
        monitor='val_dsc',
        mode='max',
        save_best_only=True)
    
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_dsc', factor=0.8, patience=2, mode = "max", verbose = 1)
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_dsc', patience=20, verbose=0, mode='max', restore_best_weights=False)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir,       
        histogram_freq=1,
        write_images=True,
        write_graph=True,
        profile_batch=0, 
    )
    #64 92%
    
    model = dense_unet(inputs, p['filters'], p['block_scale'])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=8e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
        loss={
            'lbl': happy_meal(p['alpha'], p['beta'])
            },
        metrics={
            'lbl': dsc_soft()
            }
    )

    client.load_data_in_memory()

    model.fit(x=gen_train,
        steps_per_epoch=100,
        epochs=p['epochs'],
        validation_data=gen_valid,
        validation_steps=100,
        validation_freq=1,
        callbacks=[tensorboard_callback, model_checkpoint_callback, reduce_lr_callback, early_stop_callback])
    model.save(MODEL_NAME)

def test():
    model.load('{}/ckp/'.format(p['output_dir']))

    
if __name__ == "__main__":
    train()
    
# go through xra pnuemonia model -> assess using the ground truth, they also have a ct, 
# train on ct-public then test on ct-uci