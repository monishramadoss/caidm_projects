import nibabel as nib
import numpy as np
import os
import glob
import multiprocessing
from tqdm import tqdm
from PIL import Image

def save_array(path, array, name):
    im = []
    new_array = array.copy()
    new_array = new_array / np.max(np.abs(new_array))
    new_array *= (255.0/new_array.max())
    im = [Image.fromarray(np.uint8(new_array[i]), mode='L') for i in range(new_array.shape[0])]
    im[0].save(path + name, save_all=True, append_images=im[1:])
    
if(not os.path.isdir('./image')):
    os.makedirs('./image')
    
def load_save_nii(secondary_path):
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
    data = data[idxs[0]-8:idxs[-1]+8, ...]
    label = label[idxs[0]-8:idxs[-1]+8, ...]

    label = label==3
    label = label.astype(np.float32)
    save_array('./image/', data, secondary_path.split('/')[-1]+'_data_.gif')
    save_array('./image/', label, secondary_path.split('/')[-1]+'_label_.gif')
    
    np.save(os.path.join(secondary_path, 'data_cti.npy'), data)
    np.save(os.path.join(secondary_path, 'label_r.npy'), label)
    print('done with:', secondary_path, label.shape, np.max(label), np.min(label), data.shape, np.max(data), np.min(data))

    
def load_thoracic_data(root_path: str = "./jmodels/data/Thoracic_Data") -> None:    
    pool = multiprocessing.Pool()
    pool.map(load_save_nii, [os.path.join(root_path, secondary_path) for secondary_path in os.listdir(root_path)])
    
if __name__ == '__main__':
    load_thoracic_data()