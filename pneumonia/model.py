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
    
def res_path(x, filter_count, length, name=None):
    with tf.name_scope(name+'res_path') as scope:
        s1 = layers.Conv3D(filter_count, 1, activation=None, padding='same', name=scope+"_skip_conv_1", use_bias=False, kernel_initializer='he_uniform')(x)
        s1 = layers.BatchNormalization()(s1)

        c1 = layers.Conv3D(filter_count, (1, 3, 3), padding='same', name=scope+"_conv_0", use_bias=False, kernel_initializer='he_uniform')(x)
        c1 = layers.BatchNormalization()(c1)
        c1 = layers.Activation('relu')(c1)
        
        out = layers.add([s1, c1])
        out = layers.Activation('relu')(out)
        out = layers.BatchNormalization()(out)
       
        for i in range(1,length):
            s1 = layers.Conv3D(filter_count, 1, activation=None, padding='same', use_bias=False, kernel_initializer='he_uniform')(s1)
            s1 = layers.BatchNormalization()(s1)
        
            c1 = layers.Conv3D(filter_count, (1, 3, 3), padding='same', name=scope+"_conv_"+str(i), use_bias=False, kernel_initializer='he_uniform')(out)
            c1 = layers.BatchNormalization()(c1)
            c1 = layers.Activation('relu')(c1)
        
            out = layers.add([s1, c1])
            out = layers.Activation('relu')(out)
            out = layers.BatchNormalization()(out)
        
    return out


def down_block(x, filter_count, length, name=None):
    with tf.name_scope(name) as scope:
        c1 = multires_block(x, filter_count, name=scope)
        p1 = layers.MaxPooling3D(pool_size=(1, 2, 2), name=scope+"_pool_1")(c1)
        c1 = res_path(c1, filter_count, length, name=name)
    return p1, c1

def bottleneck_block(x, filter_count, name=None):
    with tf.name_scope(name) as scope:
        c1 = multires_block(x, filter_count, name=scope)
    return c1


def up_block(x1, x2, filter_count, name=None):
    with tf.name_scope(name) as scope:
        u6 = layers.Conv3DTranspose(filter_count, (1, 2, 2), strides=(1, 2, 2), padding='same', name=scope+"_convT_1")(x1)
       # u6 = layers.BatchNormalization()(u6)
        u6 = layers.Concatenate()([u6, x2])
        c1 = multires_block(u6, filter_count, name=scope)
    return c1
    

def UNET(inputs, filters=32, size=4, fs=4):
    #encode
    e1, s1 = down_block(inputs['dat'], filters*fs, 4, 'en1')
    e2, s2 = down_block(e1, 2*filters*fs, 3, 'en2')
    e3, s3 = down_block(e2, 4*filters*fs, 2, 'en3')
    e4, s4 = down_block(e3, 8*filters*fs, 1, 'en4')
   
    #bottlneck
    b1 = bottleneck_block(e4, 16*filters*fs, 'bt1')
    #b1 = bottleneck_block(b1, 16*filters*fs, 'bt2')

    #decode
    u6 = up_block(b1, s4, 8*filters*fs, 'de1')
    u7 = up_block(u6, s3, 4*filters*fs, 'de2')
    u8 = up_block(u7, s2, 2*filters*fs, 'de3')
    u9 = up_block(u8, s1, filters*fs, 'de4')
    
    out = layers.Conv3D(2, (1,1,1), activation=tf.keras.activations.sigmoid, padding="same", use_bias=False, kernel_initializer='he_uniform')(u9)
    out = layers.BatchNormalization(name='pna')(out)
    logits = {}
    logits['pna'] = out
    return Model(inputs, logits)


def UNET3(inputs, filters=32, size=4, fs=1.5):
    x = inputs['dat']
    #encode
    e0, s0 = down_block(x, 2*filters*fs, 'x00')
    e1, s1 = down_block(e0, 4*filters*fs, 'x10')
    e2, s2 = down_block(e1, 8*filters*fs, 'x20')
    e3, s3 = down_block(e2, 16*filters*fs, 'x30')
   
    
    #bottlneck
    b1 = bottleneck_block(e3, 32*filters*fs, 'x40')
    b1 = bottleneck_block(b1, 32*filters*fs, 'x41')

    #transition layers    
    b2_6 = up_block(s3, s2, 8*filters*fs, 'x21')
    

    b1_1_12 = up_block(s2, s1, 4*filters*fs, 'x11')
    b1_2_7 = up_block(b2_6, s1+b1_1_12, 4*filters*fs, 'x12')

    
    b0_1_02 = up_block(s1, s0, 2*filters*fs, 'x01')
    b0_2_03 = up_block(b1_1_12, s0 + b0_1_02, 2*filters*fs, 'x02')
    b0_3_8 = up_block(b1_2_7, s0 + b0_2_03 + b0_1_02, 2*filters*fs, 'x03')


    #decode
    u5 = up_block(b1, s3, 16*filters*fs, 'x31')
    u6 = up_block(u5 ,s2 + b2_6, 8*filters*fs, 'x22')
    u7 = up_block(u6, s1 + b1_2_7 + b1_1_12, 4*filters*fs, 'x13')
    u8 = up_block(u7, s0 + b0_3_8 + b0_2_03 + b0_1_02, 2*filters*fs, 'x04')
    
    b5_0 = bottleneck_block(s0, 4*filters*fs, 'x50')
    b6_0 = bottleneck_block(b5_0, 8*filters*fs, 'x60')
    b7_0 = bottleneck_block(b6_0, 16*filters*fs, 'x70')
    b8_0 = bottleneck_block(b7_0, 2*filters*fs, 'x80')
    b9_0 = bottleneck_block(b8_0, 2*filters*fs, 'x90')
   
    b9 = bottleneck_block(b9_0 + u8 + b0_3_8 + b0_2_03 + b0_1_02, filters, 'xl0')
    
    logits = {}
    logits['pna'] = layers.Conv3D(2, (1,3,3), activation=tf.keras.activations.sigmoid, padding="same", name='pna')(b9)
    return Model(inputs, logits)
