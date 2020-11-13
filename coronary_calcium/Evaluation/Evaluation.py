import glob
import os

import numpy as np
import tensorflow as tf
from medpy.io import load, save
from tqdm import tqdm
from twilio.rest import Client


# import ntpath


### FUNCTIONS ###


def send_text(message):
    accountSid = 'AC0618f71f18d8b09d197ed23bcac3392d'
    authToken = 'fb8ab0729fde0c427ee43a78f18c3c7f'
    twilioClient = Client(accountSid, authToken)
    myTwilioNumber = "+19387777408"
    destCellPhone = "4252238463"
    myMessage = twilioClient.messages.create(body=message, from_=myTwilioNumber, to=destCellPhone)


def load_images(URL):
    org_strct = '/**/**.mhd'

    dictionary = {}
    mask_list = []
    image_list = []

    for file in tqdm(glob.glob(org_strct, recursive=True)):
        if 'r.mhd' in file:
            mask_list.append(file)
        elif 'cti.mhd' in file:
            image_list.append(file)

    dictionary.update({"image": image_list})
    dictionary.update({"mask": mask_list})
    return dictionary


def interumprocessing(array, mask):
    ### Add commands here ###
    return array


def volume_calculation(array, header):
    voxel_array = header.get_voxel_spacing()
    voxel_vol = voxel_array[0] * voxel_array[1] * voxel_array[2]
    Vol_sum = np.sum(np.count_nonzero(array > 0))
    volume = Vol_sum * voxel_vol
    return volume


def save_file(URL, array, volume, units):
    newURL = URL.replace("cti", "S")
    newURL = newURL.replace("***", "Evaluation")
    # Replace *** with the folder name in the URL that is "test image". This should make a new folder called evaluation at that point, but other wise
    # keep the same architecture

    save(array, newURL)

    head, tail = os.path.split(newURL)
    tail = tail.replace(".mhd", "_volume.txt")

    new2URL = os.path.join(head, tail)

    volume_file = open(new2URL, "w")
    volume_file.write("Total Patient Plaque Volume: ", volume, " mm^3")
    volume_file.close()


def make_gif():
    return None


def main(dictionary):
    print("loading models")
    Organ_seg_model = tf.keras.models.load_model('***.h5')
    lesion_seg_model = tf.keras.models.load_model('***.h5')
    # https://www.tensorflow.org/guide/keras/save_and_serialize
    # I saw you used the save model function, so we should be able to load that directly with these fucntions instead of loading weights
    print("models loaded")

    for URL in tqdm(dictionary["image"]):
        image, header = load(URL)
        placeholder = np.array()

        for slice in tqdm(image):
            mask = Organ_seg_model.predict(slice)
            slice = interumprocessing(slice, mask)
            mask = lesion_seg_model.predict(slice)
            placeholder.append(mask)

        volume = volume_calculation(placeholder, header)
        save_file(URL, placeholder, volume)
        # make gif command if you want
        del image, header, placeholder


# to keep memory low,understandable that this will increase the time


## Start of program ##

URL_dict = load_images("***")
main(URL_dict)
send_text("evaluation is complete")
