import tensorflow as tf
from tensorflow import optimizers, losses
from tensorflow.keras import Input, layers, Model

def denseunet(inputs, label, filters=32, scale_0 = 1, scale_1=1, stage_1=1, stage_2=3 , stage_3=3, stage_4=1):
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
    TD0 = conv1(filters*scale_0, inputs['dat'])
    TD1 = td_block(filters * 1, filters * 1, TD0, stage_1)
    TD2 = td_block(filters * 1.5, filters * 1, TD1, stage_1)
    TD3 = td_block(filters * 2, filters * 1.5, TD2, stage_1)
    TD4 = td_block(filters * 2.5, filters * 2, TD3, stage_2)
    
    TD5 = td_block(filters * 3, filters * 2.5, TD4, stage_2)

    TU1 = tu_block(filters * 2.5, filters * 3, TD5, TD4, stage_2, stage_3)
    TU2 = tu_block(filters * 2, filters * 2.5, TU1, TD3, stage_2, stage_3)
    TU3 = tu_block(filters * 1.5, filters * 2, TU2, TD2, stage_1, stage_4)
    TU4 = tu_block(filters * 1, filters * 1.5, TU3, TD1, stage_1, stage_4)
    TU5 = tran2(filters * scale_1, TU4)
    logits = {}
    logits[label] = layers.Conv3D(filters=2, name=label, **kwargs)(TU5)
    model = Model(inputs=inputs, outputs=logits)
    return model

