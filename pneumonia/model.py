from tensorflow.keras import  Model, layers, backend
import tensorflow as tf

def multires_block(x, filter_count, alpha=1.67, name=None):
    with tf.name_scope(name) as scope:
        W = alpha*filter_count
        f0 = int(W*0.167) + int(W*0.333) + int(W*0.5)
        f1 = int(W*0.167)
        f2 = int(W*0.333)
        f3 = int(W*0.5)

        s1 = layers.Conv3D(f0, 1, activation=None, padding='same', name=scope+"_skip_conv_1", use_bias=False, kernel_initializer='he_uniform')(x)
        s1 = layers.BatchNormalization()(s1)
        
        c1 = layers.Conv3D(f1, (1, 3, 3), padding='same', name=scope+"_conv_1", use_bias=False, kernel_initializer='he_uniform')(x)
        c1 = layers.BatchNormalization()(c1)
        c1 = layers.Activation('relu')(c1)


        c2 = layers.Conv3D(f2, (1, 3, 3), padding='same', name=scope+"_conv_2", use_bias=False, kernel_initializer='he_uniform')(c1)
        c2 = layers.BatchNormalization()(c2)
        c2 = layers.Activation('relu')(c2)

        c3 = layers.Conv3D(f3, (1, 3, 3), padding='same', name=scope+"_conv_3", use_bias=False, kernel_initializer='he_uniform')(c2)
        c3 = layers.BatchNormalization()(c3)
        c3 = layers.Activation('relu')(c3)

        cat = layers.Concatenate()([c1, c2, c3])
        cat = layers.BatchNormalization()(cat)
        out = layers.add([s1, cat])
        out = layers.Activation('relu')(out)
        out = layers.BatchNormalization()(out)
        
        
    return out
    

def dense_unet(inputs, filters=32):
    '''Model Creation'''
    #Define model#
    # Define kwargs dictionary#
    kwargs = {
    'kernel_size': (1,3,3),
    'padding': 'same',
    'bias_initializer':'zeros'} #zeros, ones, golorit_uniform
    #Define lambda functions#
    conv = lambda x, filters, strides : layers.Conv3D(filters=int(filters), strides=(1, strides, strides), **kwargs)(x)
    norm = lambda x : layers.BatchNormalization()(x)
    relu = lambda x : layers.LeakyReLU()(x)
    #Define stride-1, stride-2 blocks#
    conv1 = lambda filters, x : relu(norm(conv(x, filters, strides=1)))
    conv2 = lambda filters, x : relu(norm(conv(x, filters, strides=2)))
    #Define single transpose#
    tran = lambda x, filters, strides : layers.Conv3DTranspose(filters=int(filters), strides=(1,strides, strides), **kwargs)(x)
    #Define transpose block#
    tran2 = lambda filters, x : relu(norm(tran(x, filters, strides=2)))
    concat = lambda a, b : layers.Concatenate()([a, b])
    #Define Dense Block#
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
    #Build Model#
    # TD = convolutions that train down, DB = Dense blocks, TU = Transpose convolutions that train up, C = concatenation groups.
    
    TD1 = td_block(filters*1,filters*1, inputs['dat'],0)
    TD2 = td_block(filters*1.5,filters*1,TD1,1)
    TD3 = td_block(filters*2,filters*1.5,TD2,2)
    TD4 = td_block(filters*2.5,filters*2,TD3,3)
    TD5 = td_block(filters*3,filters*2.5,TD4,4)
    #print("TD5 shape: ", TD5.shape)
    TU1 = tu_block(filters*2.5,filters*3,TD5,TD4,4)
    TU2 = tu_block(filters*2,filters*2.5,TU1,TD3,3)
    TU3 = tu_block(filters*1.5,filters*2,TU2,TD2,2)
    TU4 = tu_block(filters*1,filters*1.5,TU3,TD1,1)
    TU5 = tran2(filters*1, TU4) 
    logits = {}
    logits['pna'] = layers.Conv3D(filters = 2, name='pna', **kwargs)(TU5)
    model = Model(inputs=inputs, outputs=logits )
    return model


def UNETpp(inputs, filters=32, size=-1, fs=-1):
    # --- Define kwargs dictionary
    kwargs = {
        'kernel_size': (1, 3, 3),
        'padding': 'same'}

    # --- Define lambda functions
    conv = lambda x, filters, strides : layers.Conv3D(filters=int(filters), strides=strides, **kwargs)(x)
    norm = lambda x : layers.BatchNormalization()(x)
    relu = lambda x : layers.LeakyReLU()(x)
    tran = lambda x, filters, strides : layers.Conv3DTranspose(filters=int(filters), strides=strides, **kwargs)(x)

    # --- Define stride-1, stride-2 blocks
    conv1 = lambda filters, x : relu(norm(conv(x, filters, strides=1)))
    conv2 = lambda filters, x : relu(norm(conv(x, filters, strides=(1, 2, 2))))
    tran2 = lambda filters, x : relu(norm(tran(x, filters, strides=(1, 2, 2))))

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
    l31  = tran2(stage_3, conv1(stage_4, tran2(stage_4, l40) + l30))
    l22  = tran2(stage_2, conv1(stage_3, l31  + l20 + l21))
    l13 =  tran2(stage_1, conv1(stage_2, l22  + l10 + l12))
    l04 =  conv1(stage_1, conv1(stage_1, l13  + l00 + l03))

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