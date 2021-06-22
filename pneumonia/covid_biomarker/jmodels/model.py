import tensorflow as tf
from tensorflow import optimizers, losses
from tensorflow.keras import Input, layers, Model

def conv_bn_relu(input, filters, kernel_size=3, stride=1, name=None):
    x = layers.Conv3D(filters, (1, kernel_size, kernel_size), padding='same', strides=(1, stride, stride), name=name, use_bias=False)(input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    return x
label = 'pna'
def unet(inputs, label, filters = 32, size=4):
    x = inputs['dat']
    encoder_block = []
    filter_start = 4
    #encoder_block
    for en in range(size):
        x = conv_bn_relu(x, filters * filter_start, name='en_'+str(en))
        x = conv_bn_relu(x, filters * filter_start)        
        x = conv_bn_relu(x, filters * filter_start, stride=2)
        filter_start = 2
        encoder_block.append(x)

    x = conv_bn_relu(x, filters * filter_start)
    x = conv_bn_relu(x, filters * filter_start)
    t1 = layers.Dense(512)(x)
    t1 = layers.Dense(512)(t1)
    for de in range(size):
        skip = encoder_block.pop(-1)
        x = layers.concatenate([skip, x])
        x = conv_bn_relu(x, filters * filter_start, name='de_'+str(de))
        x = conv_bn_relu(x, filters * filter_start)
        x = layers.UpSampling3D(size=(1,2,2))(x)
        filter_start = filter_start * 2

    logits = {}
    logits['ratio'] = layers.Dense(1, activation='sigmoid', name='ratio', use_bias=False)(t1)
    logits[label] = layers.Conv3D(2, (1,3,3), padding='same', name=label, use_bias=False)(x)
    return Model(inputs, logits)

from jarvis.train.client import Client
from jarvis.train import models, params, custom
from jarvis.utils.general import overload

try:
    from jarvis.utils.general import gpus
    gpus.autoselect(2)
except:
    pass

@overload(Client)
def preprocess(self, arrays, row, **kwargs):
    if row['corhort-uci']:
        arrays['xs']['msk-pna'][:] = 0.0
        arrays['xs']['msk-ratio'][:] = 1.0
    else:
        arrays['xs']['msk-pna'][:] = 1.0
        arrays['xs']['msk-ratio'][:] = 0.0
    arrays['xs']['msk-pna'] = arrays['xs']['msk-pna'] > 0
    return arrays

# --- Create a test Client
client = Client('/data/raw/covid_biomarker/data/ymls/client-dual-256.yml')
gen_train_all, gen_valid = client.create_generators()
inputs = client.get_inputs(Input)
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir)

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_dsc_1', factor=0.8, patience=2, mode="max",
                                                            verbose=1)
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_dsc_1', patience=5, verbose=0, mode='max',
                                                        restore_best_weights=False)

model = unet(inputs, 32)
#model = da_unet(inputs)
print(model.summary())
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss={ 'lbl': happy_meal(None, p['alpha'], p['beta'])},
    metrics={ 'lbl': custom.dsc() },
    experimental_run_tf_function=False
)

client.load_data_in_memory()

model.fit(
    x=gen_train,
    epochs=40,
    steps_per_epoch=1000,
    validation_data=gen_valid,
    validation_steps=500,
    validation_freq=1,
    callbacks=[model_checkpoint_callback, reduce_lr_callback, early_stop_callback]
)

model.save(MODEL_NAME, overwrite=True, include_optimizer=False)
