
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from padding_images import *
from PIL import Image


image_list=os.listdir('/home/capture/Downloads/fire-dataset-dunnings/superpixels/isolated-superpixels/train/fire')

image=Image.open('/home/capture/Downloads/fire-dataset-dunnings/superpixels/isolated-superpixels/train/fire'+'/'+'barbecue0sp0.png')
imageBox = image.getbbox()
cropped=image.crop(imageBox)
np_image=np.array(cropped)
#print(np_image.shape)
output_image_size=resize_image(np_image,[224,224,3])
print(output_image_size.shape)
img=Image.fromarray(output_image_size,'RGB')
img.show()
