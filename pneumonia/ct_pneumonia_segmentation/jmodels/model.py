# -*- coding: utf-8 -*-
"""cadim pneumonia pna RAUNET

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_rjmMTLespZytwl5bFMe-ao7lI--OzeQ
"""

import datetime
# Commented out IPython magic to ensure Python compatibility.
# % pip install -q jarvis-md
# % pip install -q tensorflow-model-optimizationpi
import os

import numpy as np
from numpy.core.arrayprint import StructuredVoidFormat
import tensorflow as tf
from tensorflow import optimizers, losses
from tensorflow.keras import Input, layers, Model

# import tensorflow_addons as tfa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.compat.v1.disable_eager_execution()
from jarvis.train import datasets, params, custom
from jarvis.train.client import Client
from jarvis.utils.general import overload #gpus

#gpus.autoselect(1)
def UNETpp(inputs, filters=32, size=-1, fs=-1):
    # --- Define kwargs dictionary
    kwargs = {
        'kernel_size': (1, 3, 3),
        'padding': 'same',
        'use_bias': False}

    # --- Define lambda functions
    conv = lambda x, filters, strides: layers.Conv3D(filters=int(filters), strides=strides, **kwargs)(x)
    norm = lambda x: layers.BatchNormalization()(x)
    relu = lambda x: layers.LeakyReLU()(x)
    tran = lambda x, filters, strides: layers.Conv3DTranspose(filters=int(filters), strides=strides, **kwargs)(x)

    # --- Define stride-1, stride-2 blocks
    conv1 = lambda filters, x: relu(norm(conv(x, filters, strides=1)))
    conv2 = lambda filters, x: relu(norm(conv(x, filters, strides=(1, 2, 2))))
    tran2 = lambda filters, x: relu(norm(tran(x, filters, strides=(1, 2, 2))))

    # --- Define pooling operations
    # apool = lambda x, strides : layers.AveragePooling3D(pool_size=(1, 3, 3), strides=strides, padding='same')(x)
    # mpool = lambda x, strides : layers.MaxPooling3D(pool_size=(1, 3, 3), strides=strides, padding='same')(x)
    # avgpool = lambda x, strides : relu(norm(apool(x, strides)))
    # maxpool = lambda x, strides : relu(norm(mpool(x, strides)))

    stage_1 = filters
    stage_2 = filters * 1.5
    stage_3 = filters * 2
    stage_4 = filters * 2.5
    stage_5 = filters * 3

    l00 = conv1(stage_1, conv1(stage_1, inputs['dat']))
    l10 = conv1(stage_2, conv1(stage_2, conv2(stage_2, l00)))
    l20 = conv1(stage_3, conv1(stage_3, conv2(stage_3, l10)))
    l30 = conv1(stage_4, conv1(stage_4, conv2(stage_4, l20)))
    l40 = conv1(stage_5, conv1(stage_5, conv2(stage_5, l30)))

    # --- Inner bottom layer
    l21 = tran2(stage_3, conv1(stage_3, conv2(stage_3, l20 + tran2(stage_3, l30))))

    # --- Inner middle layer
    l11 = conv1(stage_2, conv1(stage_2, conv1(stage_2, l10 + tran2(stage_2, l20))))
    l12 = conv1(stage_2, conv1(stage_2, conv1(stage_2, l11 + l10 + tran2(stage_2, l21))))

    # --- Inner top layer
    l01 = conv1(stage_1, conv1(stage_1, conv1(stage_1, l00 + tran2(stage_1, l10))))
    l02 = conv1(stage_1, conv1(stage_3, conv1(stage_1, l01 + l00 + tran2(stage_1, l11))))
    l03 = conv1(stage_1, conv1(stage_4, conv1(stage_1, l02 + l01 + l00 + tran2(stage_1, l12))))

    # --- Outer expanding layers
    l31 = tran2(stage_3, conv1(stage_4, tran2(stage_4, l40) + l30))
    l22 = tran2(stage_2, conv1(stage_3, l31 + l20 + l21))
    l13 = tran2(stage_1, conv1(stage_2, l22 + l10 + l12))
    l04 = conv1(stage_1, conv1(stage_1, l13 + l00 + l03))

    # --- Deep CNN
    l50 = conv1(stage_2, conv1(stage_2, conv1(stage_2, l00)))
    l60 = conv1(stage_3, conv1(stage_3, conv1(stage_3, l50)))
    l70 = conv1(stage_4, conv1(stage_4, conv1(stage_4, l60)))
    lstage_4 = conv1(stage_1, conv1(stage_1, conv1(stage_1, l70)))
    l90 = conv1(stage_1, conv1(stage_1, conv1(stage_1, lstage_4)))

    # Grouping output layer
    loss_layer = conv1(stage_1, conv1(stage_1, l01 + l02 + l03 + l04 + l90))

    # --- Create logits
    logits = {}
    logits['pna'] = layers.Conv3D(filters=2, name='pna', **kwargs)(loss_layer)
    return Model(inputs, logits)

def dense_unet(inputs, filters=64):
    '''Model Creation'''
    # Define model#
    # Define kwargs dictionary#
    kwargs = {
        'kernel_size': (1, 3, 3),
        'padding': 'same',
        'use_bias': False}  # zeros, ones, golorit_uniform
    # Define lambda functions#
    conv = lambda x, filters, strides: layers.Conv3D(filters=int(filters), strides=(1, strides, strides), **kwargs)(x)
    norm = lambda x: layers.BatchNormalization()(x)
    relu = lambda x: layers.LeakyReLU()(x)
    # Define stride-1, stride-2 blocks#
    conv1 = lambda filters, x: relu(norm(conv(x, filters, strides=1)))
    conv2 = lambda filters, x: relu(norm(conv(x, filters, strides=2)))
    # Define single transpose#
    tran = lambda x, filters, strides: layers.Conv3DTranspose(filters=int(filters), strides=(1, strides, strides),
                                                              **kwargs)(x)
    # Define transpose block#
    tran2 = lambda filters, x: relu(norm(tran(x, filters, strides=2)))
    concat = lambda a, b: layers.Concatenate()([a, b])

    # Define Dense Block#
    def dense_block(filters, input, DB_depth):
        ext = 3 + DB_depth
        outside_layer = input
        for _ in range(0, int(ext)):
            inside_layer = conv1(filters, outside_layer)
            outside_layer = concat(outside_layer, inside_layer)
        return outside_layer

    def td_block(conv1_filters, conv2_filters, input, DB_depth):
        TD = conv1(conv1_filters, conv2(conv2_filters, input))
        DB = dense_block(conv1_filters, TD, DB_depth)
        return DB

    def tu_block(conv1_filters, tran2_filters, input, td_input, DB_depth, skip_DB_depth=1):
        t1 = tran2(tran2_filters, input)
        TU = dense_block(conv1_filters, t1, skip_DB_depth)
        C = concat(TU, td_input)
        DB = dense_block(conv1_filters, C, DB_depth)
        return DB

    # Build Model#
    # TD = convolutions that train down, DB = Dense blocks, TU = Transpose convolutions that train up, C = concatenation groups.

    TD1 = td_block(filters * 1, filters * 1, inputs['dat'], 1)
    TD2 = td_block(filters * 1.5, filters * 1, TD1, 1)
    TD3 = td_block(filters * 2, filters * 1.5, TD2, 1)
    TD4 = td_block(filters * 2.5, filters * 2, TD3, 3)
    TD5 = td_block(filters * 3, filters * 2.5, TD4, 3)

    TU1 = tu_block(filters * 2.5, filters * 3, TD5, TD4, 3, 3)
    TU2 = tu_block(filters * 2, filters * 2.5, TU1, TD3, 3, 3)
    TU3 = tu_block(filters * 1.5, filters * 2, TU2, TD2, 1, 1)
    TU4 = tu_block(filters * 1, filters * 1.5, TU3, TD1, 1, 1)
    TU5 = tran2(filters * 1, TU4)
    logits = {}
    logits['pna'] = layers.Conv3D(filters=2, name='pna', **kwargs)(TU5)
    model = Model(inputs=inputs, outputs=logits)
    return model

def da_unet(inputs, filters=48, fs=1):
    kwargs = {
        'kernel_size': (1, 3, 3),
        'padding': 'same',
        'use_bias': False
    }  # zeros, ones, golorit_uniform
    # Define lambda functions#
    conv = lambda x, filters, strides: layers.Conv3D(filters=int(filters), strides=(1, strides, strides), **kwargs)(x)
    norm = lambda x: layers.BatchNormalization()(x)
    relu = lambda x: layers.LeakyReLU()(x)
    bneck = lambda x, filters, strides: layers.Conv3D(filters=int(filters), strides=(1, strides, strides), kernel_size=1, padding='same', use_bias=False)

    # Define stride-1, stride-2 blocks#
    conv1 = lambda filters, x: relu(norm(conv(x, filters, strides=1)))
    conv2 = lambda filters, x: relu(norm(conv(x, filters, strides=2)))
    # Define single transpose#
    tran = lambda x, filters, strides: layers.Conv3DTranspose(filters=int(filters), strides=(1, strides, strides),
                                                              **kwargs)(x)
    # Define transpose block#
    tran2 = lambda filters, x: relu(norm(tran(x, filters, strides=2)))
    concat = lambda a, b: layers.Concatenate()([a, b])
    upsample = lambda x, size: layers.UpSampling3D(size=size)(x)

    # Define Dense Block#
    def dense_block(filters, input, DB_depth):
        ext = 3 + DB_depth
        outside_layer = input
        for _ in range(0, int(ext)):
            inside_layer = conv1(filters, outside_layer)
            outside_layer = concat(outside_layer, inside_layer)
        return outside_layer

    def td_block(conv1_filters, conv2_filters, input, DB_depth):
        TD = conv1(conv1_filters, conv2(conv2_filters, input))
        DB = dense_block(conv1_filters, TD, DB_depth)
        return DB

    def tu_block(conv1_filters, tran2_filters, input, td_input, DB_depth, skip_DB_depth=1):
        t1 = tran2(tran2_filters, input)
        TU = dense_block(conv1_filters, t1, skip_DB_depth)
        C = concat(TU, td_input)
        DB = dense_block(conv1_filters, C, DB_depth)
        return DB

    # Build Model #
    # TD = convolutions that train down, DB = Dense blocks, TU = Transpose convolutions that train up, C = concatenation groups.
    kernel_initializer=tf.keras.initializers.VarianceScaling()
    bias_initializer = tf.keras.initializers.Zeros()

    def channel_attention(x,  ratio):
        channel = x.get_shape()[-1]
        avg_pool = tf.reduce_mean(x, axis=[1,2,3], keepdims=True)
        avg_pool = layers.Dense(channel//ratio, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(avg_pool)
        avg_pool = layers.Dense(channel, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(avg_pool)
        
        max_pool = tf.reduce_max(x, axis=[1,2,3], keepdims=True)
        max_pool = layers.Dense(channel//ratio, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(max_pool)
        max_pool = layers.Dense(channel, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(max_pool)
        
        scale = layers.Activation('sigmoid')(avg_pool + max_pool)
        return x * scale

    def spatial_attention(x):
        kernel_size = 7
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], -1)
        concat = layers.Conv3D(1, kernel_size=[1, kernel_size, kernel_size], padding='same', use_bias=False, kernel_initializer=kernel_initializer, activation='sigmoid')(concat)
        return x * concat

    def cbam_block(x, ratio=4):
        attention_feature = channel_attention(x,  ratio)
        attention_feature = spatial_attention(attention_feature )
        return x

    TD1 = td_block(filters * 1, filters * 1, inputs['dat'], 1)
    TD2 = td_block(filters * 1.5, filters * 1, TD1, 1)
    TD3 = td_block(filters * 2, filters * 1.5, TD2, 1)
    TD4 = td_block(filters * 2.5, filters * 2, TD3, 3)
    TD5 = td_block(filters * 3, filters * 2.5, TD4, 3)

    A4 = cbam_block(x=TD4)
    TU1 = tu_block(filters * 2.5, filters * 3, TD5, A4 , 3, 3)
    
    A3 = cbam_block(x=TD3)
    TU2 = tu_block(filters * 2, filters * 2.5, TU1, A3, 3, 3)

    A2 = cbam_block(x=TD2)
    TU3 = tu_block(filters * 1.5, filters * 2, TU2, A2, 1, 1)

    A1 = cbam_block(x=TD1)
    TU4 = tu_block(filters * 1.0, filters * 1.5, TU3, A1, 1, 1)

    TU5 = tran2(filters * 1, TU4)

    logits = {}
    logits['pna'] = layers.Conv3D(filters=2, name='pna', **kwargs)(TU5)
    
    model = Model(inputs=inputs, outputs=logits)
    return model



@overload(Client)
def preprocess(self, arrays, **kwargs):
    msk = np.zeros(arrays['xs']['dat'].shape)
    lng = arrays['xs']['lng'] > 0
    pna = arrays['ys']['pna'] > 0
    msk[lng] = 1.0
    # msk[pna] = 10.0
    # arrays['ys']['pna'][pna] = 1.0
    arrays['xs']['lng'] = msk
    arrays['xs']['dat'] *= p['negative']
    return arrays


paths = datasets.download(name='ct/pna')
p = params.load(csv='./hyper.csv', row=0)
configs = {'batch': {'size': p['batch_size'], 'fold': 0}}
MODEL_NAME = '{}/ckp/model.h5'.format(p['output_dir'])
path = '{}/data/ymls/client.yml'.format(paths['code'])
client = Client(path, configs=configs)
gen_train, gen_valid = client.create_generators()


# path2 = '{}/data/ymls/client-uci.yml'.format(paths['code'])
# client2 = Client(path2, configs={'batch': {'size': 1, 'fold': -1}})
# gen_valid, _ = client2.create_generators()

inputs = client.get_inputs(Input)

log_dir = "{}/logs/".format(p['output_dir']) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if (not os.path.isdir(log_dir)):
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



def train():
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='{}/ckp/'.format(p['output_dir']),
        monitor='val_dsc_1',
        mode='max',
        save_best_only=True)

    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir)

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_dsc_1', factor=0.8, patience=2, mode="max",
                                                              verbose=1)
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_dsc_1', patience=5, verbose=0, mode='max',
                                                           restore_best_weights=False)

    model = dense_unet(inputs, 64)
    #model = da_unet(inputs)
    print(model.summary())
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss={ 'pna': happy_meal(inputs['lng'], p['alpha'], p['beta'])},
        metrics={ 'pna': custom.dsc(weights=inputs['lng']) },
        experimental_run_tf_function=False
    )

    client.load_data_in_memory()

    model.fit(
        x=gen_train,
        epochs=40,
        steps_per_epoch=1000,
        validation_data=gen_valid,
        validation_steps=500,
        validation_freq=1,
        callbacks=[model_checkpoint_callback, reduce_lr_callback, early_stop_callback]
    )

    model.save(MODEL_NAME, overwrite=True, include_optimizer=False)
    return model

def test(model):
    paths = datasets.download(name='ct/pna')
    client = Client('{}/data/ymls/client.yml'.format(paths['code']), configs={'batch': {'fold': 0}})

    def dice(y_true, y_pred, c=1, epsilon=1):
        true = y_true[..., 0] == c
        pred = np.argmax(y_pred, axis=-1) == c 

        A = np.count_nonzero(true & pred) * 2
        B = np.count_nonzero(true) + np.count_nonzero(pred) + epsilon
        
        return A / B


    _, test_valid = client.create_generators(test=True)
    dsc = []

    for x, y in test_valid:
        logits = model.predict(x)
        if type(logits) is dict:
            logits = logits['pna'] * x['lng']
        dsc.append(dice(y['pna'][0], logits[0]))
        
    # --- Create array
    dsc = np.array(dsc)
    print('Dice, mean: {}'.format(np.mean(dsc)))
    print('Dice, median: {}'.format(np.median(dsc)))

m = train()
#test(m)

