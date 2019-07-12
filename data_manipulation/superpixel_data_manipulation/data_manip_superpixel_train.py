'''
data manipulation for superpixels
centering isolated superpixels and saving it in hdf5 format.
'''

import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from padding_images import *
from PIL import Image
import h5py
import random

directory='/home/pbu/Downloads/fire-dataset-dunnings/superpixels/isolated-superpixels/train/'
no_fire_directory='{}nofire'.format(directory)
fire_directory='{}fire'.format(directory)
output_directory='/home/pbu/Desktop/ganesh-samarth-fire-detection-copy'



dataset=list()
for items in os.listdir(fire_directory):

    items_list=list()
    items_list.append(items)
    items_list.append(1)
    dataset.append(items_list)

for items in random.sample(os.listdir(no_fire_directory),9000):
    items_list=list()
    items_list.append(items)
    items_list.append(0)
    dataset.append(items_list)

for items in os.listdir('/home/pbu/Downloads/nofirered-dataset-ganesh-superpixels'):
    items_list = list()
    items_list.append(items)
    items_list.append(0)
    dataset.append(items_list)


print(len(os.listdir(fire_directory)))
print(len(os.listdir(no_fire_directory)))
random.shuffle(dataset)


X = []
Y = []

count = 0

for line in dataset:
    path = line[0]
    label = line[1]
    if label == 1:
        image = Image.open(fire_directory+'/'+path)
        imageBox = image.getbbox()
        cropped = image.crop(imageBox)
        np_image = np.array(cropped)
        #print(np_image.shape)
        output_image_size=resize_image(np_image,[112,112,3])
        label = [1,0]
        #print(1)
    else:
        if os.path.exists(no_fire_directory+'/'+path):
            image = Image.open(no_fire_directory+'/'+path)
        else:
            image=Image.open('/home/pbu/Downloads/nofirered-dataset-ganesh-superpixels/'+path)
        imageBox = image.getbbox()
        cropped = image.crop(imageBox)
        np_image = np.array(cropped)
        #print(np_image.shape)
        output_image_size=resize_image(np_image,[112,112,3])
        label = [0,1]
        #print(0)
    X.append(output_image_size)
    Y.append(label)
    count += 1
    print(count)

    

# Create hdf5 dataset

h5f = h5py.File("{}data_superpixel_with_super_red_centre_nonaugmented_112.h5".format(output_directory), 'w')
h5f.create_dataset("X", data=X)
h5f.create_dataset("Y", data=Y)
h5f.close()

