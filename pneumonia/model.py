from tensorflow.keras import  Model, layers, backend

def residual_block(input, input_channels=None, output_channels=None, kernel_size=(1, 3, 3), stride=1, name='out'):
    """
    full pre-activation residual block
    https://arxiv.org/pdf/1603.05027.pdf
    """
    if output_channels is None:
        output_channels = input.get_shape()[-1]
    if input_channels is None:
        input_channels = output_channels // 4

    strides = (1, stride, stride)

    

    x = layers.BatchNormalization()(input)
    x = layers.Activation('relu')(x)
    x = layers.Conv3D(input_channels, (1, 1, 1), use_bias=False)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv3D(input_channels, kernel_size, padding='same', strides=stride, use_bias=False)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv3D(output_channels, (1, 1, 1), padding='same', use_bias=False)(x)

    if input_channels != output_channels or stride != 1:
        input = layers.Conv3D(output_channels, (1, 1, 1), padding='same', strides=strides, use_bias=False)(input)
    if name == 'out':
        x = layers.add([x, input])
    else:
        x = layers.add([x, input], name=name)
    return x

def attention_block(input, input_channels=None, output_channels=None, encoder_depth=1, name='out'):
    """
    attention block
    https://arxiv.org/abs/1704.06904
    """
    p = 3
    t = 6
    r = 3

    if input_channels is None:
        input_channels = input.get_shape()[-1]
    if output_channels is None:
        output_channels = input_channels
    for i in range(p):
        input = residual_block(input)

    output_trunk = input
    for i in range(t):
        output_trunk = residual_block(output_trunk, output_channels=output_channels)

    output_soft_mask = layers.MaxPooling3D(pool_size=(1,2,2), padding='same')(input) 

    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)

    skip_connections = []
    for i in range(encoder_depth - 1):
        output_skip_connection = residual_block(output_soft_mask)
        skip_connections.append(output_skip_connection)
        output_soft_mask = layers.MaxPooling3D(pool_size=(1,2,2), padding='same')(output_soft_mask)
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)
    skip_connections = list(reversed(skip_connections))

    for i in range(encoder_depth - 1):
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)
        output_soft_mask = layers.UpSampling3D(size=(1,2,2))(output_soft_mask)
        output_soft_mask = layers.add([output_soft_mask, skip_connections[i]])

    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)

    output_soft_mask = layers.UpSampling3D(size=(1,2,2))(output_soft_mask)
    output_soft_mask = layers.Conv3D(output_trunk.get_shape()[-1], (1, 1, 1), use_bias=False)(output_soft_mask)
    output_soft_mask = layers.Conv3D(output_trunk.get_shape()[-1], (1, 1, 1), use_bias=False)(output_soft_mask)
    output_soft_mask = layers.Activation('sigmoid')(output_soft_mask)
  

    # Attention: (1 + output_soft_mask) * output_trunk
    output = layers.Lambda(lambda x: x + 1)(output_soft_mask)
    output = layers.Multiply()([output, output_trunk])  #

    # Last Residual Block
    for i in range(p):
        output = residual_block(output, name=name + str(i))

    return output

def build_res_atten_unet_3d(x, filter_num=32, merge_axis=-1, pool_size=(1, 2, 2)
                            , up_size=(1, 2, 2)):
    data = x
    conv1 = layers.Conv3D(filter_num * 4, 3, padding='same', use_bias=False)(data)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)

    pool = layers.MaxPooling3D(pool_size=pool_size)(conv1)

    res1 = residual_block(pool, output_channels=filter_num * 4)

    pool1 = layers.MaxPooling3D(pool_size=pool_size)(res1)

    res2 = residual_block(pool1, output_channels=filter_num * 8)

    pool2 = layers.MaxPooling3D(pool_size=pool_size)(res2)

    res3 = residual_block(pool2, output_channels=filter_num * 16)
    pool3 = layers.MaxPooling3D(pool_size=pool_size)(res3)

    res4 = residual_block(pool3, output_channels=filter_num * 32)

    pool4 = layers.MaxPooling3D(pool_size=pool_size)(res4)

    res5 = residual_block(pool4, output_channels=filter_num * 64)
    res5 = residual_block(res5, output_channels=filter_num * 64)

    atb5 = attention_block(res4, encoder_depth=1, name='atten1')
    up1 = layers.UpSampling3D(size=up_size)(res5)
    merged1 = layers.concatenate([up1, atb5], axis=merge_axis)

    res5 = residual_block(merged1, output_channels=filter_num * 32)

    atb6 = attention_block(res3, encoder_depth=2, name='atten2')
    up2 = layers.UpSampling3D(size=up_size)(res5)
    merged2 = layers.concatenate([up2, atb6], axis=merge_axis)

    res6 = residual_block(merged2, output_channels=filter_num * 16)
    atb7 = attention_block(res2, encoder_depth=3, name='atten3')
    up3 = layers.UpSampling3D(size=up_size)(res6)
    merged3 = layers.concatenate([up3, atb7], axis=merge_axis)

    res7 = residual_block(merged3, output_channels=filter_num * 8)
    atb8 = attention_block(res1, encoder_depth=4, name='atten4')
    up4 = layers.UpSampling3D(size=up_size)(res7)
    merged4 = layers.concatenate([up4, atb8], axis=merge_axis)

    res8 = residual_block(merged4, output_channels=filter_num * 4)
    up = layers.UpSampling3D(size=up_size)(res8)
    merged = layers.concatenate([up, conv1], axis=merge_axis)
    conv9 = layers.Conv3D(filter_num * 4, (1,3,3), padding='same', use_bias=False)(merged)
    conv9 = layers.BatchNormalization()(conv9)
    conv9 = layers.Activation('relu')(conv9)
    return conv9

def RA_UNET(inputs):
    x = backend.abs(layers.Multiply()([inputs['dat'], inputs['lng']]))
    x = build_res_atten_unet_3d(x)
    logits = {}
    logits['pna'] = layers.Conv3D(2, (1,3,3), padding="same", name='pna', use_bias=False)(x)
    return Model(inputs, logits)
    