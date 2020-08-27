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
        x = conv_bn_relu(x, filters * filter_start, stride=2)
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
