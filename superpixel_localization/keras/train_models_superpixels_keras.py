'''
Training models for superpixel data in keras

'''

from resnet_keras import *
from inception_resnet_v1_keras import *
from inception_resnet_v2_keras import *
from efficientnet_keras import *
#import tensorflow as tf
from keras.callbacks import ModelCheckpoint
#from evaluate_in_train_keras import *
import h5py
import sys


model_name=sys.argv[1]
output_directory='/home/pbu/Desktop/ganesh-samarth-fire-detection-copy'
h5f = h5py.File("{}data_superpixel_with_red_centre_nonaugmented.h5".format(output_directory), 'r')
train_X = h5f['X']
train_Y = h5f['Y']
h5f1 = h5py.File("{}data_superpixel_test_centre.h5".format(output_directory), 'r')
test_X = h5f1['X']
test_Y = h5f1['Y']
input_shape=[224,224,3]
model=globals()[model_name](nb_classes=2)
total_parameters = 0


checkpointer = ModelCheckpoint(filepath=model_name+"_keras.h5", verbose=1, save_best_only=True)
model.compile('rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])


model.fit(train_X,train_Y,batch_size=64,epochs=30,validation_data=(test_X,test_Y),callbacks=[checkpointer],shuffle='batch')
#evaluate_in_train_keras(model_name,model,i)

h5f.close()
