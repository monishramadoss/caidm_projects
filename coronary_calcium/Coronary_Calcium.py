import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import optimizers #, losses
from tensorflow.keras import Input
#import tensorflow_addons as tfa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from jarvis.train import datasets
from jarvis.train.client import Client
from jarvis.utils.general import overload, tools as jtools

from model import RA_UNET

paths = datasets.download(name='ct/structseg')

# full body seg
# client = Client('{}/data/ymls/client-full.yml'.format(paths['code']))
client = Client('{}/data/ymls/client-cardiac.yml'.format(paths['code']))
gen_train, gen_valid = client.create_generators()

def dice(y_true, y_pred, c=1, epsilon=1):
    A = 0
    B = 0
    y_true_slice = y_true
    y_pred_slice = y_pred
    true = y_true_slice[..., 0] == c
    pred = np.argmax(y_pred_slice, axis=-1) == c
    A = np.count_nonzero(true & pred) * 2
    B = np.count_nonzero(true) + np.count_nonzero(pred) + epsilon
    return A / B

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.cast(tf.keras.backend.flatten(tf.keras.backend.argmax(y_pred)), y_true_f.dtype)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
    client.load_data_in_memory()

def train():
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./ckp/',
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    model = RA_UNET(inputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=2e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
        loss={
            'pna': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            },
        metrics={
            'pna': ['accuracy', dice_coef]
        })
    model.fit(x=gen_train,
        steps_per_epoch=300,
        epochs=30,
        validation_data=gen_valid,
        validation_steps=100,
        validation_freq=10,
        callbacks=[tensorboard_callback, model_checkpoint_callback])

def test(model):
    lung_seg = []
    for x,y in test_valid:
        logits = model.predict(x)
        if type(logits) is dict:
            logits = logits['pna']
        lung_seg.append(dice(y['pna'][0], logits[0]))

    lung_seg = np.array(lung_seg)
    print(lung_seg.mean())

if __name__ == "__main__":
    train()