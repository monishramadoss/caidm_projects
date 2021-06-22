import tensorflow as tf
from tensorflow import optimizers, losses
from tensorflow.keras import Input, layers, Model

def unetpp(inputs, label, filters=32, block_1=1.0, block_2=1.5, block_3=2.0, block_4=2.5, block_5=3.0):
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

    stage_1 = filters * block_1
    stage_2 = filters * block_2
    stage_3 = filters * block_3
    stage_4 = filters * block_4
    stage_5 = filters * block_5

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
    logits[label] = layers.Conv3D(filters=2, name=label, **kwargs)(loss_layer)
    return Model(inputs, logits)


