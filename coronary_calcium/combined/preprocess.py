import glob
import multiprocessing
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from medpy.io import load


def save_array(path, array, name):
    fig = plt.figure()
    ims = [[plt.imshow(array[i, ...], animated=True)] for i in range(array.shape[0])]
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(os.path.join(path, name))


if (not os.path.isdir('./image')):
    os.makedirs('./image')


def load_save_nii(secondary_path, idx=0):
    data_path = os.path.join(secondary_path, 'data.nii.gz')
    label_path = os.path.join(secondary_path, 'label.nii.gz')
    data = nib.load(data_path).get_fdata()
    label = nib.load(label_path).get_fdata()
    data = np.transpose(np.rot90(data, 3), (2, 0, 1))
    label = np.transpose(np.rot90(label, 3), (2, 0, 1))
    idxs = []
    for i in range(data.shape[0]):
        label_match = np.count_nonzero(label[i] == 3)
        if label_match > 0:
            idxs.append(i)

    sz = 12
    data = data[idxs[0] - sz:idxs[-1] + sz, ...]
    label = label[idxs[0] - sz:idxs[-1] + sz, ...]
    data = np.flip(data, axis=(2))
    label = np.flip(label, axis=(2))
    label = label == 3
    label = label.astype(np.float32)
    save_array('./image/', data, secondary_path.split('/')[-1] + '_data_.gif')
    save_array('./image/', label, secondary_path.split('/')[-1] + '_label_.gif')

    np.save(os.path.join(secondary_path, 'data_cti.npy'), data)
    np.save(os.path.join(secondary_path, 'label_r.npy'), label)
    print('done with:', secondary_path, label.shape, np.max(label), np.min(label), data.shape, np.max(data),
          np.min(data))


def load_save_mhd(secondary_path, idx=0):
    misc_name = glob.glob1(secondary_path, '*ctai.mhd')[0]
    data_name = glob.glob1(secondary_path, '*cti.mhd')[0]
    # label_name = glob.glob1(secondary_path, '*r.mhd')[0]
    misc_path = os.path.join(secondary_path, misc_name)
    data_path = os.path.join(secondary_path, data_name)
    # label_path = os.path.join(secondary_path, label_name)

    data, data_header = load(data_path)
    # label, label_header = load(label_path)
    misc, misc_header = load(misc_path)

    data = np.transpose(np.rot90(data, 3), (2, 0, 1))
    # label = np.transpose(np.rot90(label, 3), (2, 0, 1))
    misc = np.transpose(np.rot90(misc, 3), (2, 0, 1))

    # label[label!=0] = 1

    #np.save(os.path.join(secondary_path, 'data_ctai.npy'), data)
    #np.save(os.path.join(secondary_path, ''))
    # print(data.shape, label.shape, misc.shape)
    #save_array('./image', data, 'data_{}.gif'.format(idx))
    # save_array('./image', label, 'label_{}.gif'.format(idx))
    #save_array('./image', misc, 'misc_{}.gif'.format(idx))
    print('done with:', secondary_path, label.shape, np.max(label), np.min(label), data.shape, np.max(data),
          np.min(data))

def load_thoracic_data(root_path: str = "./jmodels/data/Thoracic_Data") -> None:
    with multiprocessing.Pool(24) as pool:
        pool.map(load_save_nii, [os.path.join(root_path, secondary_path) for i, secondary_path in enumerate(os.listdir(root_path))])
    

def load_plaque_data(root_path: str = "./jmodels/data/Plaque_Data/Training_Set") -> None:
    root_path = os.path.join(root_path, 'Train')
    dirs = [os.path.join(root_path, secondary_path) for i, secondary_path in enumerate(os.listdir(root_path))]
    with multiprocessing.Pool(24) as pool:
        pool.map(load_save_mhd, dirs)

if __name__ == '__main__':
    load_thoracic_data()
    load_plaque_data()
