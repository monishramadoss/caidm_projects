import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import optimizers #, losses
from tensorflow.keras import Input
#import tensorflow_addons as tfa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from jarvis.train import datasets, custom
from jarvis.train.client import Client
from jarvis.utils.general import overload, tools as jtools

from model import *

paths = datasets.download(name='ct/structseg')
client = Client('./client-heart.yml'.format(paths['code']))
gen_train, gen_valid = client.create_generators()
inputs = client.get_inputs(Input)

x, y = next(gen_train)


def train():
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./ckp/',
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    model = dense_unet_andrew(inputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=2e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
        loss={
            'lbl': custom.sce(weights=None)
            },
        metrics={
            'lbl': ['accuracy', custom.dsc()]
        })
    model.fit(x=gen_train,
        steps_per_epoch=300,
        epochs=50,
        validation_data=gen_valid,
        validation_steps=100,
        validation_freq=10,
        callbacks=[tensorboard_callback, model_checkpoint_callback])

def test(model):
    lung_seg = []
    for x,y in test_valid:
        logits = model.predict(x)
        if type(logits) is dict:
            logits = logits['lbl']
        lung_seg.append(dice(y['lbl'][0], logits[0]))

    lung_seg = np.array(lung_seg)
    print(lung_seg.mean())

if __name__ == "__main__":
    train()
    pass