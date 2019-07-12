'''
Tflearn train models 
'''

from tflearn_firenet import *
from vgg13 import *
import tensorflow as tf
from tflearn_inceptionv1onfire import *
from tflearn_inceptionv2onfire_a import *
from tflearn_inceptionv2onfire_b import *
from tflearn_inceptionv2onfire_c import *
from inceptionv2_tflearn_a_4 import *
from inceptionv2_tflearn_a_5 import *
from inceptionv2_tflearn_a_6 import *
from inceptionv2_tflearn_b_4 import *
from inceptionv2_tflearn_b_5 import *
from inceptionv2_tflearn_b_6 import *
from inceptionv2_tflearn_c_4 import *
from inceptionv2_tflearn_c_5 import *
from inceptionv2_tflearn_c_6 import *
from inceptionv2_tflearn_a import *
from inceptionv2_tflearn_b import *
from inceptionv2_tflearn_c import *
from inceptionv3_tflearn import *
from inceptionv3_a_tflearn import *
from inceptionv3_b_tflearn import *
from inceptionv3_c_tflearn import *
from inceptionv3_d_tflearn import *
from inceptionv3_e_tflearn import *
from inceptionv3_f_tflearn import *
from inceptionv3_g_tflearn import *
from inceptionv3_h_tflearn import *
from inceptionv3_i_tflearn import *
from inceptionv3_j_tflearn import *
from inceptionv3_k_tflearn import *
from inceptionv3_l_tflearn import *
from inceptionv4_tflearn import *
from inceptionv4_b_tflearn import *
from inceptionv4_c_tflearn import *
from inceptionv4_d_tflearn import *
from inceptionv4_e_tflearn import *
from inceptionv4_f_tflearn import *
from inceptionv4_g_tflearn import *
from inceptionv4_h_tflearn import *
from inceptionv4_i_tflearn import *
from inceptionv4_j_tflearn import *
from inceptionv4_k_tflearn import *
from inceptionv4_l_tflearn import *
from inceptionv4_m_tflearn import *


from evaluate_in_train_superpixel_tflearn import *
import h5py
import sys
import os

model_name=globals()['construct_'+sys.argv[1]]

output_directory='/home/pbu/Desktop/ganesh-samarth-fire-detection-copy'


if sys.argv[2]=='lrn':
	normalization='local_response_normalization'
else:
	normalization='batch_normalization'


if sys.argv[3]=='rmsprop':
	optimizer='rmsprop'
	num_epochs=50
else:
	optimizer='momentum'
	num_epochs=50




if sys.argv[4]=='d':
	dropout=0.4
else:
	dropout=0

if sys.argv[5]=='prelu':
	activation='prelu'
else:
	activation='relu'



model = model_name(224, 224,normalization,optimizer,dropout,activation,training=True)
print("Constructed ..."+sys.argv[1])


h5f1 = h5py.File("{}data_superpixel_test_centre.h5".format(output_directory), 'r')
final_test_accuracy=list()
final_f_score=list()
final_precision=list()
final_tpr=list()
final_fnr=list()
test_X = h5f1['X']
test_Y = h5f1['Y']

h5f = h5py.File("{}data_superpixel_with_red_centre_nonaugmented.h5".format(output_directory), 'r')
train_X = h5f['X']
train_Y = h5f['Y']
total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    #print(shape)
    #print(len(shape))
    variable_parameters = 1
    for dim in shape:
        #print(dim)
        variable_parameters *= dim.value
    #print(variable_parameters)
    total_parameters += variable_parameters
print(total_parameters)



for i in range(0,150):
	
	model.fit(train_X,train_Y,n_epoch=1,validation_set=(test_X,test_Y),batch_size=64,show_metric=True,shuffle=True)
	evaluate_in_train_superpixel(sys.argv[1],model,i)
	




with open('parameters_count.txt','a') as myfile2:
	myfile2.write('models_new_weights_superpixel/'+'centre_nonaugmented_with_red/'+sys.argv[1]+'/'+sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3]+'_'+str(sys.argv[4])+sys.argv[5]+'_sp'+'.tflearn'+'\n')
	myfile2.write(str(total_parameters)+'\n')


