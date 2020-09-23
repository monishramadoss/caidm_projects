import os
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow import optimizers, losses
from tensorflow.keras import Input
#import tensorflow_addons as tfa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from jarvis.train import datasets, custom
from jarvis.train.client import Client
from jarvis.utils.general import overload, tools as jtools

from model import *

paths = datasets.download(name='ct/structseg')

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
        return l2(y_true, y_pred) + beta - l1(y_true, y_pred)
    return calc_loss

client = Client('./client-heart.yml')
gen_train, gen_valid = client.create_generators()
inputs = client.get_inputs(Input)
@overload(Client)
def preprocess(self, arrays, **kwargs):   
    arrays['ys']['lbl'] = arrays['ys']['lbl'] 
    msk = cv2.GaussianBlur(arrays['ys']['lbl'], (99,99), 0 )
    return arrays

def train():
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./ckp/',
        save_weights_only=True,
        monitor='val_dcs',
        mode='max',
        save_best_only=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    model = andrew_NN(inputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=2e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
        loss={
            'lbl': happy_meal(1.0, 0.5)
            },
        metrics={
            'lbl': dsc_soft()
            }
    )
    
    model.fit(x=gen_train,
        steps_per_epoch=600,
        epochs=50,
        validation_data=gen_valid,
        validation_steps=100,
        validation_freq=1,
        callbacks=[tensorboard_callback, model_checkpoint_callback])

if __name__ == "__main__":
    train()
    pass