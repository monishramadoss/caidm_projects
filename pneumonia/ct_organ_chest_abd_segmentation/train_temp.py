import os
from tensorflow import keras as K
import tensorflow as tf
from tensorflow import optimizers, losses
from tensorflow.keras import Input, layers, Model

import kerastuner as kt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from jarvis.train import datasets, params, custom
from jarvis.train.client import Client
from jarvis.utils.general import overload 
try:
    from jarvis.utils.general import gpus
    gpus.autoselect(1)

except:
    pass


paths = datasets.download(name='ct/organ-chest-abd')
configs = {'batch': {'size': 8, 'fold': 0}}

path = '{}/data/ymls/client-lung.yml'.format(paths['code'])
client = Client(path, configs=configs)
client.load_data_in_memory()
gen_train, gen_valid = client.create_generators()

xs, ys = next(gen_train)
for key, arr in xs.items():
    print('xs key: {} | shape = {}'.format(key.ljust(8), arr.shape))
for key, arr in ys.items():
    print('ys key: {} | shape = {}'.format(key.ljust(8), arr.shape))

inputs = client.get_inputs(Input)
labels = 'lbl'

%(model)s


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
        return (1.0 - A / B) * scale
    return dsc

def sce(weights=None, scale=1.0):
    loss = losses.SparseCategoricalCrossentropy(from_logits=True)
    @tf.function
    def sce(y_true, y_pred):
        return loss(y_true=y_true, y_pred=y_pred, sample_weight=weights) * scale
    return sce

def happy_meal(weights=None, alpha=5, beta=1,  epsilon=0.01, cls=1):
    l2 = sce(None, alpha)
    l1 = dsc_soft(weights, beta, epsilon, cls)
    @tf.function
    def calc_loss(y_true, y_pred):
        return l2(y_true, y_pred) + l1(y_true, y_pred)
    return calc_loss


def build_model(hp):
    model = %(define)s
    alpha = hp.Choice('alpha', [0.3, 1.0])
    beta = hp.Choice('beta', [0.3, 1.0])
    
    model.compile(
        optimizer='adam',
        loss={ labels: happy_meal(None, alpha, beta)},
        metrics={ labels: custom.dsc() },
        experimental_run_tf_function=False
    )
    return model

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_dsc_1', factor=0.8, patience=2, mode="max", verbose=0)
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_dsc_1', patience=5, verbose=0, mode='max', restore_best_weights=False)
objective = kt.Objective("val_dsc_1", direction="max")
tuner = kt.Hyperband(
    build_model,
    objective = objective, 
    max_epochs = 100,     
    factor = 3,
    hyperband_iterations = 1,
    directory = "%(dis)s",
    project_name = "%(project_name)s",
    overwrite = True
)

tuner.search(
    x=gen_train,
    epochs=40,
    steps_per_epoch=1000,
    validation_data=gen_valid,
    validation_steps=500,
    validation_freq=1,
    callbacks=[reduce_lr_callback, early_stop_callback]
)