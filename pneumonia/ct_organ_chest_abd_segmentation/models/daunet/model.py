import tensorflow as tf
from tensorflow import optimizers, losses
from tensorflow.keras import Input, layers, Model

def daunet(inputs, label, filters=32, scale_0 = 1, scale_1=1, stage_1=1, stage_2=3 , stage_3=3, stage_4=1):
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

    TD0 = conv1(filters*scale_0, inputs['dat'])
    TD1 = td_block(filters * 1, filters * 1, TD0, stage_1)
    TD2 = td_block(filters * 1.5, filters * 1, TD1, stage_1)
    TD3 = td_block(filters * 2, filters * 1.5, TD2, stage_1)
    TD4 = td_block(filters * 2.5, filters * 2, TD3, stage_2)
    
    TD5 = td_block(filters * 3, filters * 2.5, TD4, stage_2)    

    A2 = cbam_block(x=TD2)
    A1 = cbam_block(x=TD1)
    A4 = cbam_block(x=TD4)
    A3 = cbam_block(x=TD3)

    TU1 = tu_block(filters * 2.5, filters * 3, TD5, A4, stage_2, stage_3)
    TU2 = tu_block(filters * 2, filters * 2.5, TU1, A3, stage_2, stage_3)
    TU3 = tu_block(filters * 1.5, filters * 2, TU2, A2, stage_1, stage_4)
    TU4 = tu_block(filters * 1, filters * 1.5, TU3, A1, stage_1, stage_4)
    TU5 = tran2(filters * scale_1, TU4)


    TU5 = tran2(filters * 1, TU4)

    logits = {}
    logits[label] = layers.Conv3D(filters=2, name=label, **kwargs)(TU5)
    
    model = Model(inputs=inputs, outputs=logits)
    return model

