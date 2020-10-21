
import os, glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import optimizers, losses
from tqdm import tqdm, trange
import datetime
from tensorflow.keras import Input, layers, Model, callbacks
from tensorflow.keras.utils import Sequence
from jarvis.train import datasets, custom, params
from jarvis.utils.general import overload, tools as jtools, gpus
gpus.autoselect(1)
#import tensorflow_addons as tfa


from PIL import Image
def save_array(path, array, name):
    im = []
    new_array = array.copy()
    new_array = new_array / np.max(np.abs(new_array),axis=0)
    new_array *= (255.0/new_array.max())

    im = [Image.fromarray(np.uint8(new_array[i]), mode='L') for i in range(new_array.shape[0])]
    im[0].save(path + name, save_all=True, append_images=im[1:])
    
p = params.load(csv='./hyper.csv', row=0)
os.makedirs(p['output_dir'], exist_ok=True)
MODEL_NAME = '{}/ckp/model.hdf5'.format(p['output_dir'])
log_dir = "{}/logs/".format(p['output_dir']) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

if(not os.path.isdir(log_dir)):
    os.makedirs(log_dir)
    
if(not os.path.isdir('./image')):
    os.makedirs('./image')

def data_process(data_path='./data/CAIDM_Data'.format(p['output_dir']), batch_size=4, train_percent=0.9, cls=(1,2,3)):
    gz_files = list()
    data_array = None
    label_array = None
    count = 0
    data_paths = os.listdir(data_path)
    for f in trange(len(data_paths), desc='processing data files'):
        file = data_paths[f]
        main_path = os.path.join(data_path, file)
        d_path = glob.glob(os.path.join(main_path, "*cti.npy"))[0]
        l_path = glob.glob(os.path.join(main_path, "*r.npy"))[0]
        gz_files.append([d_path, l_path])
        data = np.load(d_path)
        label = np.load(l_path)   
        data = data[1:-1]
        label = data[1:-1]
        
        save_array('./image/', data, file+'_data_.gif')
        save_array('./image/', label, file+'_label_.gif')
        
        data, label = np.expand_dims(data, (1)), np.expand_dims(data, (1))
        zero_count = np.count_nonzero(label)
        count += label.shape[0]
        if zero_count > 0:
            for i in range(data.shape[0]):
                data_array = data[i] if data_array is None else np.concatenate([data_array, data[i]])               
                label[label != 0] = 1                    
                label[label == 0] = 0
                label_array = label[i] if label_array is None else np.concatenate([label_array, label[i]])

    data_array = np.expand_dims(data_array, (1, -1)).astype(np.float32)
    data_array = np.clip(data_array, -1024, 400)
    data_array /= 200
    label_array = np.expand_dims(label_array, (1, -1))
    sz = data_array.shape[0]
    train_dataset = tf.data.Dataset.from_tensor_slices((data_array[:int(sz*train_percent)], label_array[:int(sz*train_percent)])).shuffle(100).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((data_array[int(sz*train_percent):], label_array[int(sz*train_percent):])).batch(1)
    return train_dataset, test_dataset


def dense_unet(inputs, filters=32):
    '''Model Creation'''
    # Define kwargs dictionary
    kwargs = {
        'kernel_size': (1,3,3),
        'padding': 'same',
        'bias_initializer':'zeros'
    } 
    # Define lambda functions#
    conv = lambda x, filters, strides : layers.Conv3D(filters=int(filters), strides=(1, strides, strides), **kwargs)(x)
    norm = lambda x : layers.BatchNormalization()(x)
    relu = lambda x : layers.LeakyReLU()(x)
    # Define stride-1, stride-2 blocks#
    conv1 = lambda filters, x : relu(norm(conv(x, filters, strides=1)))
    conv2 = lambda filters, x : relu(norm(conv(x, filters, strides=2)))
    # Define single transpose#
    tran = lambda x, filters, strides : layers.Conv3DTranspose(filters=int(filters), strides=(1,strides, strides), **kwargs)(x)
    # Define transpose block#
    tran2 = lambda filters, x : relu(norm(tran(x, filters, strides=2)))
    concat = lambda a, b : layers.Concatenate()([a, b])
    # Define Dense Block#
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
        
    TD1 = td_block(filters*1,filters*1, inputs ,0)
    TD2 = td_block(filters*1.5,filters*1,TD1,1)
    TD3 = td_block(filters*2,filters*1.5,TD2,2)
    TD4 = td_block(filters*2.5,filters*2,TD3,3)
    TD5 = td_block(filters*3,filters*2.5,TD4,4)
    
    TU1 = tu_block(filters*2.5,filters*3,TD5,TD4,4)
    TU2 = tu_block(filters*2,filters*2.5,TU1,TD3,3)
    TU3 = tu_block(filters*1.5,filters*2,TU2,TD2,2)
    TU4 = tu_block(filters*1,filters*1.5,TU3,TD1,1)
    TU5 = tran2(filters*1, TU4) 
    logits = {}
    logits = layers.Conv3D(filters=2, **kwargs)(TU5)
    model = Model(inputs=inputs, outputs=logits )
    return model


def dsc_soft(weights=None, scale=1.0, epsilon=0.01, cls=1):
    @tf.function
    def dsc(y_true, y_pred):
        true = tf.cast(y_true[..., 0] == 1, tf.float32)
        pred = tf.nn.softmax(y_pred, axis=-1)[..., 1]
        if weights is not None:
            true = true * (weights[...]) 
            pred = pred * (weights[...])
        A = tf.math.reduce_sum(true * pred) * 2
        B = tf.math.reduce_sum(true) + tf.math.reduce_sum(pred) + epsilon
        return  (A / B) * scale
    def dsc_2(y_true, y_pred):
        true = tf.cast(y_true[..., 0] == 2, tf.float32)
        pred = tf.nn.softmax(y_pred, axis=-1)[..., 2]
        if weights is not None:
            true = true * (weights[...]) 
            pred = pred * (weights[...])
        A = tf.math.reduce_sum(true * pred) * 2
        B = tf.math.reduce_sum(true) + tf.math.reduce_sum(pred) + epsilon
        return  (A / B) * scale
    def dsc_3(y_true, y_pred):
        true = tf.cast(y_true[..., 0] == 3, tf.float32)
        pred = tf.nn.softmax(y_pred, axis=-1)[..., 3]
        if weights is not None:
            true = true * (weights[...]) 
            pred = pred * (weights[...])
        A = tf.math.reduce_sum(true * pred) * 2
        B = tf.math.reduce_sum(true) + tf.math.reduce_sum(pred) + epsilon
        return  (A / B) * scale
    if cls == 1:
        return dsc
    elif cls == 2:
        return dsc_2
    else:
        return dsc_3


def sce(weights=None, scale=1.0):
    loss = losses.SparseCategoricalCrossentropy(from_logits=True)
    @tf.function
    def sce(y_true, y_pred):
        return loss(y_true=y_true, y_pred=y_pred, sample_weight=weights) * scale
    return sce


def happy_meal(alpha = 5, beta = 1, weights=None, epsilon=0.01, cls=1):
    #0 = sce(weights, alpha)
    l1 = dsc_soft(weights, beta, epsilon, 1)
    
    @tf.function
    def calc_loss(y_true, y_pred):
        return - l1(y_true, y_pred) 
    return calc_loss


def train():
    gen_train, gen_valid = data_process()

    loaded_seg_model = dense_unet(Input(shape=(1, 512, 512, 1)), 64)
    loaded_seg_model.load('./model.hdf5')
    logits = loaded_seg_model.evaluate(gen_valid)
    
    
    model_checkpoint_callback = callbacks.ModelCheckpoint(filepath='./{}/ckp/'.format(p['output_dir']),
        save_weights_only=True,
        monitor='val_dsc',
        mode='max',
        save_best_only=True)
    
    reduce_lr_callback = callbacks.ReduceLROnPlateau(monitor='dsc', factor=0.8, patience=2, mode = "max", verbose = 1)
    early_stop_callback = callbacks.EarlyStopping(monitor='val_dsc', patience=20, verbose=0, mode='max', restore_best_weights=False)
    tensorboard_callback = callbacks.TensorBoard(log_dir,       
        histogram_freq=1,
        write_images=True,
        write_graph=True,
        profile_batch=0, 
    )
    #64 92%

    model = dense_unet(Input(shape=(1, 512, 512, 1)), p['filters'])
    model.compile(optimizer=optimizers.Adam(learning_rate=8e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
        loss= happy_meal(p['alpha'], p['beta']),
        metrics=[dsc_soft(cls=1)]
    )
    
    model.fit(x=gen_train,
        epochs=p['epochs'],
        validation_data=gen_valid,
        validation_freq=1,
        callbacks=[tensorboard_callback, model_checkpoint_callback, reduce_lr_callback, early_stop_callback])

    model.save(MODEL_NAME)


if __name__ == "__main__":
    train()