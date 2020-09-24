from tensorflow.keras import  Model, layers, backend

def conv_bn_relu(input, filters, kernel_size=3, stride=1, name=None):
    x = layers.Conv3D(filters, (1, kernel_size, kernel_size), padding='same', strides=(1, stride, stride), name=name)(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def UNET(inputs, filters = 32, size=4):
    x = inputs['dat']

    encoder_block = []
    filter_start = 4
    #encoder_block
    for en in range(size):
        x = conv_bn_relu(x, filters * filter_start, name='en_'+str(en))
        x = conv_bn_relu(x, filters * filter_start)        
        x = layers.MaxPooling3D(pool_size=(1,2,2))(x)
        filter_start *= 2
        encoder_block.append(x)

    x = conv_bn_relu(x, filters * filter_start)
    x = conv_bn_relu(x, filters * filter_start)

    for de in range(size):
        skip = encoder_block.pop(-1)
        x = layers.concatenate([skip, x])
        x = conv_bn_relu(x, filters * filter_start, name='de_'+str(de))
        x = conv_bn_relu(x, filters * filter_start)
        x = layers.UpSampling3D(size=(1,2,2))(x)
        filter_start = filter_start // 2

    logits = {}
    logits['lbl'] = layers.Conv3D(2, (1,3,3), padding="same", name='lbl', use_bias=False)(x)
    return Model(inputs, logits)


def dense_unet_andrew(inputs):
    '''Model Creation'''

    #Define model#
    # Define kwargs dictionary#
    kwargs = {
    'kernel_size': (1,3,3),
    'padding': 'same',
    } #zeros, ones, golorit_uniform

    #Define lambda functions#
    conv = lambda x, filters, strides : layers.Conv3D(filters=filters, strides=(1, strides,strides), **kwargs)(x)
    norm = lambda x : layers.BatchNormalization()(x)
    relu = lambda x : layers.LeakyReLU()(x)

    #Define stride-1, stride-2 blocks#
    conv1 = lambda filters, x : relu(norm(conv(x, filters, strides=1)))
    conv2 = lambda filters, x : relu(norm(conv(x, filters, strides=2)))

    #Define single transpose#
    tran = lambda x, filters, strides : layers.Conv3DTranspose(filters=filters, strides=(1, strides, strides), **kwargs)(x)

    #Define transpose block#
    tran2 = lambda filters, x : relu(norm(tran(x, filters, strides=2)))
    concat = lambda a, b : layers.Concatenate()([a, b])

    #Define Dense Block#
    def dense_block(filters,input):
        l1 = conv1(filters, input)
        c1 = concat(input,l1)
        l2 = conv1(filters,c1)
        c2 = concat(c1,l2)
        l3 = conv1(filters, c2)
        c3 = concat(c2,l3)
        l4 = conv1(filters,c3)
        return concat(input,l4)

    def td_block(conv1_filters,conv2_filters,input):
        TD = conv1(conv1_filters,conv2(conv2_filters,input))
        DB = dense_block(conv1_filters,TD)
        return DB

    def tu_block(conv1_filters,tran2_filters,input,td_input):
        TU = conv1(conv1_filters,tran2(tran2_filters,input))
        C = concat(TU,td_input)
        DB = dense_block(conv1_filters,C)
        return DB

    #Build Model#
    # TD = convolutions that train down, DB = Dense blocks, TU = Transpose convolutions that train up, C = concatenation groups.

    filter_count = 32
    TD1 = conv2(filter_count,inputs['dat']) #basic convolution of stride 2, no maxpool.
    DB1 = dense_block(filter_count,TD1)
    
    TD2 = td_block(filter_count*2,filter_count, DB1)
    TD3 = td_block(filter_count*4,filter_count*2,TD2)
    TD4 = td_block(filter_count*8,filter_count*4,TD3)
    TD5 = td_block(filter_count*16,filter_count*8,TD4)
   
    TU1 = tu_block(filter_count*8,filter_count*16,TD5,TD4)
    TU2 = tu_block(filter_count*4,filter_count*8,TU1,TD3)
    TU3 = tu_block(filter_count*2,filter_count*4,TU2,TD2)
    TU4 = tu_block(filter_count,filter_count*2,TU3,TD1)
    TU5 = tran2(filter_count, TU4)

    logits = {}
    logits['lbl'] = layers.Conv3D(filters=2, name='lbl', **kwargs)(TU5)
    model = Model(inputs=inputs, outputs=logits)

    return model

def andrew_NN(inputs):
    '''Model Creation'''
    #Define model#
    # Define kwargs dictionary#
    kwargs = {
    'kernel_size': (1,3,3),
    'padding': 'same',
    'bias_initializer':'zeros'} #zeros, ones, golorit_uniform
    #Define lambda functions#
    conv = lambda x, filters, strides : layers.Conv3D(filters=filters, strides=(1, strides, strides), **kwargs)(x)
    norm = lambda x : layers.BatchNormalization()(x)
    relu = lambda x : layers.LeakyReLU()(x)
    #Define stride-1, stride-2 blocks#
    conv1 = lambda filters, x : relu(norm(conv(x, filters, strides=1)))
    conv2 = lambda filters, x : relu(norm(conv(x, filters, strides=2)))
    #Define single transpose#
    tran = lambda x, filters, strides : layers.Conv3DTranspose(filters=filters, strides=(1,strides, strides), **kwargs)(x)
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
    
    TD1 = td_block(10,10, inputs['dat'],0)
    TD2 = td_block(15,10,TD1,1)
    TD3 = td_block(20,15,TD2,2)
    TD4 = td_block(25,20,TD3,3)
    TD5 = td_block(30,25,TD4,4)
    #print("TD5 shape: ", TD5.shape)
    TU1 = tu_block(25,30,TD5,TD4,4)
    TU2 = tu_block(20,25,TU1,TD3,3)
    TU3 = tu_block(15,20,TU2,TD2,2)
    TU4 = tu_block(10,15,TU3,TD1,1)
    TU5 = tran2(1, TU4) 
    logits = {}
    logits['lbl'] = layers.Conv3D(filters = 2, name='lbl', **kwargs)(TU5)
    model = Model(inputs=inputs, outputs=logits )
    return model


def andrew_NN_2(inputs):
    '''Model Creation'''
    #Define model#
    # Define kwargs dictionary#
    kwargs = {
    'kernel_size': (1,3,3),
    'padding': 'same',
    'bias_initializer':'zeros'} #zeros, ones, golorit_uniform
    #Define lambda functions#
    conv = lambda x, filters, strides : layers.Conv3D(filters=filters, strides=(1, strides, strides), **kwargs)(x)
    norm = lambda x : layers.BatchNormalization()(x)
    relu = lambda x : layers.ReLU()(x)
    #Define stride-1, stride-2 blocks#
    conv1 = lambda filters, x : relu(norm(conv(x, filters, strides=1)))
    conv2 = lambda filters, x : relu(norm(conv(x, filters, strides=2)))
    #Define single transpose#
    tran = lambda x, filters, strides : layers.Conv3DTranspose(filters=filters, strides=(1,strides, strides), **kwargs)(x)
    #Define transpose block#
    tran2 = lambda filters, x : relu(norm(tran(x, filters, strides=2)))
    concat = lambda a, b : layers.Concatenate()([a, b])
    #Define Dense Block#
    def dense_block(filters,input,DB_depth):
        outside_layer = input
        for _ in range(0,int(DB_depth)):
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
    '''print("Input Shape: ", inputs.shape)
                TD1 = conv2(10,inputs) #basic convolution of stride 2, no maxpool.
                print("TD Shape: ", TD1.shape)
                DB1 = dense_block(10,TD1)
                print("Dense Block Shape: ",DB1.shape )'''
    TD1 = td_block(48,32, inputs['dat'],2)
    TD2 = td_block(72,48,TD1,4)
    TD3 = td_block(88,72,TD2,6)
    TD4 = td_block(104,88,TD3,8)
    TD5 = td_block(120,104,TD4,10)
    #print("TD5 shape: ", TD5.shape)
    TU1 = tu_block(104,120,TD5,TD4,8)
    TU2 = tu_block(88,104,TU1,TD3,6)
    TU3 = tu_block(72,88,TU2,TD2,4)
    TU4 = tu_block(48,72,TU3,TD1,2)
    TU5 = tran2(32, TU4) 
    logits = {}
    logits['lbl'] = layers.Conv3D(filters = 2, name='lbl',  **kwargs)(TU5)
    model = Model(inputs=inputs, outputs=logits )
    return model