import datetime
import glob
import os
import shutil
import warnings
from multiprocessing import Pool, freeze_support

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from scipy import ndimage
from tensorflow import optimizers, losses
from tensorflow.keras import Input, Model, layers

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if os.path.isdir('./image'):
    shutil.rmtree('./image')
    os.makedirs('./image')

if os.name == "nt":
    freeze_support()

from jarvis.train import params, datasets
from jarvis.train.client import Client

try:
    from jarvis.utils.general import gpus
    gpus.autoselect(1)
except:
    pass

paths = datasets.download(name='ct/structseg', path='./data/StructSeg_Data')
p = params.load(csv='./hyper.csv', row=1)
configs = {
    'batch': {'size': p['batch_size'], 'fold': p['fold']},
    'specs': {
        'xs': {'dat': {'shape': [1, 512, 512, 1]}},
        'ys': {'lbl': {'shape': [1, 512, 512, 1]}}
    }}
path = '{}/data/ymls/client-heart.yml'.format(paths['code'])

client = Client(path, configs=configs)
gen_train, gen_valid = client.create_generators()
inputs = client.get_inputs(Input)

os.makedirs(p['output_dir'], exist_ok=True)
log_dir = "{}/logs/".format(p['output_dir']) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def save_array(path, array, name):
    fig = plt.figure()
    ims = [[plt.imshow(array[i, 0, ...], animated=True)] for i in range(array.shape[0])]
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(os.path.join(path, name))


if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

t = np.linspace(-10, 10, 30)
bump = np.exp(-0.1 * t ** 2)
kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
kernel = np.expand_dims(kernel, (0, -1))
struct1 = ndimage.generate_binary_structure(4, 1)


def dilate(a, iter_count):
    a = np.expand_dims(a, 1)
    a = ndimage.binary_dilation(a, structure=struct1, iterations=iter_count)
    a = np.squeeze(a, 1)
    return a


# Data
def plaque_transform(cls=-1, model=None):
    def transform(data, label):
        data = data[1:-1]
        label = label[1:-1]
        label = dilate(label, 2)
        data = np.clip(data, -1024, 256) / 128
        label[label != 0] = 1
        data, label = np.expand_dims(data, 1), np.expand_dims(label, 1)
        return data, label
    return transform


def thoracic_transform():
    crop = 512
    zoom = 3
    center_x = 60
    center_y = 45

    def transform(data, label):
        min = np.min(data)

        shape = data.shape[-2]
        start = shape // 2 - crop // 2
        data = ndimage.interpolation.zoom(data, zoom, order=2)
        label = ndimage.interpolation.zoom(label, zoom, order=2)
        print(data.shape, label.shape)
        data = data[:, start - center_x:start + crop - center_x, start - center_y:start + crop - center_y]
        label = label[:, start - center_x:start + crop - center_x, start - center_y:start + crop - center_y]

        lx, ly = data.shape[-2], data.shape[-1]
        X, Y = np.ogrid[0:lx, 0:ly]
        mask = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > lx * ly / 4
        data[:, mask] = min
        label[:, mask] = 0
        

        data = np.clip(data, -1024, 256) / 128
        label = np.clip(label, 0, 1)
        
        data, label = np.expand_dims(data, 1), np.expand_dims(label, 1)
        return data, label

    return transform


def pre_process_func(data_path, file, transform, save_images):
    main_path = os.path.join(data_path, file)
    d_path = glob.glob(os.path.join(main_path, "*cti.npy"))[0]
    l_path = glob.glob(os.path.join(main_path, "*r.npy"))[0]
    if transform == 'thoracic':
        transform_fn = thoracic_transform()
    else:
        transform_fn = plaque_transform()

    data, label = transform_fn(np.load(d_path), np.load(l_path))
    if np.count_nonzero(label) > 0:
        if save_images:
            save_dir = os.path.join('./image/{0}/'.format(data_path.split('/')[-1]))
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            save_array(save_dir, data, file + '_data_.gif')
            save_array(save_dir, label, file + '_label_.gif')
            print("done with: {}".format(main_path))

    return data, label


def pre_process_func_wrapper(args):
    return pre_process_func(*args)


def data_process(data_path, batch_size=4, transform=None, train_percent=0.8, save_images=True):
    pool_args = [(data_path, f, transform, save_images) for f in os.listdir(data_path)]
    with Pool(22) as pool:
        result = pool.map(pre_process_func_wrapper, pool_args)

    data_array = result[0][0]
    label_array = result[0][1]
    for x, y in result:
        if x is not None or y is not None:
            data_array = np.concatenate([data_array, x])
            label_array = np.concatenate([label_array, y])

    data_array = np.expand_dims(data_array, (-1)).astype(np.float32)
    label_array = np.expand_dims(label_array, (-1)).astype(int)

    assert (data_array.shape == label_array.shape)
    sz = data_array.shape[0]

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (data_array[:int(sz * train_percent)], label_array[:int(sz * train_percent)])
    ).shuffle(100).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (data_array[int(sz * train_percent):], label_array[int(sz * train_percent):])
    ).batch(1)

    return train_dataset, test_dataset,


# MODEL
def dense_unet(inputs, filters=32, fs=1):
    '''Model Creation'''
    # Define kwargs dictionary
    kwargs = {
        'kernel_size': (1, 3, 3),
        'padding': 'same',
        'bias_initializer': 'zeros'
    }
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
        ext = 4 + DB_depth
        outside_layer = input
        for _ in range(0, int(ext)):
            inside_layer = conv1(filters, outside_layer)
            outside_layer = concat(outside_layer, inside_layer)
        return outside_layer

    def td_block(conv1_filters, conv2_filters, input, DB_depth):
        TD = conv1(conv1_filters, conv2(conv2_filters, input))
        DB = dense_block(conv1_filters, TD, DB_depth)
        return DB

    def tu_block(conv1_filters, tran2_filters, input, td_input, DB_depth):
        TU = conv1(conv1_filters, tran2(tran2_filters, input))
        C = concat(TU, td_input)
        DB = dense_block(conv1_filters, C, DB_depth)
        return DB

    # Build Model#
    # TD = convolutions that train down, DB = Dense blocks, TU = Transpose convolutions that train up, C = concatenation groups.

    TD1 = td_block(filters * 1, filters * 1, inputs['dat'], 0 * fs)
    TD2 = td_block(filters * 1.5, filters * 1, TD1, 1 * fs)
    TD3 = td_block(filters * 2, filters * 1.5, TD2, 2 * fs)
    TD4 = td_block(filters * 2.5, filters * 2, TD3, 3 * fs)
    TD5 = td_block(filters * 3, filters * 2.5, TD4, 4 * fs)

    TU1 = tu_block(filters * 2.5, filters * 3, TD5, TD4, 4 * fs)
    TU2 = tu_block(filters * 2, filters * 2.5, TU1, TD3, 3 * fs)
    TU3 = tu_block(filters * 1.5, filters * 2, TU2, TD2, 2 * fs)
    TU4 = tu_block(filters * 1, filters * 1.5, TU3, TD1, 1 * fs)
    TU5 = tran2(filters * 1, TU4)
    logits = {}
    logits['lbl'] = layers.Conv3D(filters=2, name='lbl', **kwargs)(TU5)
    model = Model(inputs=inputs, outputs=logits)
    return model


# LOSS
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
        return (A / B) * scale

    return dsc


def sce(weights=None, scale=1.0):
    loss = losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def sce(y_true, y_pred):
        return loss(y_true=y_true, y_pred=y_pred, sample_weight=weights) * scale

    return sce


def happy_meal(alpha=5, beta=1, weights=None, epsilon=0.01, cls=1):
    l2 = sce(weights, alpha)
    l1 = dsc_soft(weights, beta, epsilon, cls)

    @tf.function
    def calc_loss(y_true, y_pred):
        return l2(y_true, y_pred) - l1(y_true, y_pred)

    return calc_loss


# TRAIN LOOP
def _train(train_data, test_data, x, epochs, filters, block_scale, alpha, beta, checkpoint_path):
    CHECKPOINT_PATH = os.path.join(p['output_dir'], checkpoint_path)

    if not os.path.isdir(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                                                   save_weights_only=True,
                                                                   monitor='val_dsc',
                                                                   mode='max',
                                                                   save_best_only=True)

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_dsc', factor=0.8, patience=2, mode="max",
                                                              verbose=1)
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_dsc', patience=20, verbose=0, mode='max',
                                                           restore_best_weights=False)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1, write_images=True,
                                                          write_graph=True, profile_batch=0, )

    model = dense_unet(x, filters, block_scale)
    if os.path.isfile(CHECKPOINT_PATH + "/{}_model.hdf5".format(checkpoint_path)):
        model.load_weights(CHECKPOINT_PATH + "/{}_model.hdf5".format(checkpoint_path))
    else:
        model.compile(
            optimizer=optimizers.Adam(learning_rate=8e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
            loss={'lbl': happy_meal(alpha, beta)},
            metrics={'lbl': dsc_soft()}
        )

        model.fit(x=train_data,
                  epochs=epochs,
                  steps_per_epoch=500,
                  validation_data=test_data,
                  validation_freq=1,
                  validation_steps=100,
                  callbacks=[tensorboard_callback, model_checkpoint_callback, reduce_lr_callback, early_stop_callback])

        model.save(CHECKPOINT_PATH + "/{}_model.hdf5".format(checkpoint_path))
    return model


def _eval(model=None, data=[], name="", max=100):
    eval_path = os.path.join(p['output_dir'], 'images', name)
    print('Writing Out To: ' + eval_path)

    if not os.path.isdir(eval_path):
        os.makedirs(eval_path, exist_ok=True)
    avg = 0
    mini = 0 
    maxi = 0

    for i, d in tqdm(enumerate(data), total=max):
        if model is not None:
            logits = model.predict(d)
            if type(d) is dict:
                d = d['dat']
            if type(logits) is dict:
                logits = logits['lbl']
            avg += np.mean(d)
            mini += np.min(d)
            maxi += np.max(d)

        for b in range(d[0].shape[0]):
            if model is not None:
                fig = plt.figure()
                pred = tf.squeeze(tf.math.argmax(logits[b], axis=-1))
                img = plt.imshow(pred)
                fig.savefig('{0}/prediction_{1}_{2}.png'.format(eval_path, i, b))
            fig = plt.figure()
            data = tf.squeeze(d[0][b])
            img = plt.imshow(data)
            fig.savefig('{0}/input_{1}_{2}.png'.format(eval_path, i, b))
        if i >= max:
            break
    print("AVG: {0}, MAX: {1}, MIN: {2}".format(avg/max, mini/max, maxi/max))


def train():
    plaque_train, plaque_test = data_process('./data/Plaque_Data', batch_size=p['batch_size'],
                                             transform='plaque', save_images=True)

    thoracic_train, thoracic_test = data_process('./data/Thoracic_Data', batch_size=p['batch_size'],
                                                 transform='thoracic', save_images=True)

    # thoracic_model = _train(gen_train, gen_valid, inputs, 40, p['filters1'], p['block_scale1'], p['alpha'], p['beta'], 'ckp_1')
 
    thoracic_model = _train(thoracic_train, thoracic_test, {'dat': Input(shape=(1, 512, 512, 1))}, 40,
                            p['filters1'], p['block_scale1'], p['alpha'], p['beta'], 'ckp_1')
 
    _eval(thoracic_model, plaque_train, 'plaque_train')
    _eval(thoracic_model, plaque_test, 'plaque_test')
    _eval(thoracic_model, gen_valid, 'heart_test')

    # plaque_model = _train(plaque_train, plaque_test, {'dat': Input(shape=(1, 512, 512, 1))}, p['epochs'],
    #                       p['filters2'], p['block_scale2'], p['gamma'], p['delta'], 'ckp_2')


if __name__ == "__main__":
    train()
