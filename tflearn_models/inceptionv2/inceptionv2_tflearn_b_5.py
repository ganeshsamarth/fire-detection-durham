from __future__ import division, print_function, absolute_import

import tflearn
import h5py
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression


def construct_inceptionv2_b_5(x,y,normalization_method,optimizer,dropout_value,activation,training=False):

    # Build network
    # 227 x 227 original size
    network = input_data(shape=[None, y, x, 3])
    conv1_7_7 = conv_2d(network, 64, 7, strides=2, activation=activation, name = 'conv1_7_7_s2')
    pool1_3_3 = max_pool_2d(conv1_7_7, 3,strides=2)
    pool1_3_3 = globals()[normalization_method](pool1_3_3)
    conv2_3_3_reduce = conv_2d(pool1_3_3, 64,1, activation=activation,name = 'conv2_3_3_reduce')
    conv2_3_3 = conv_2d(conv2_3_3_reduce, 192,3, activation=activation, name='conv2_3_3')
    conv2_3_3 = globals()[normalization_method](conv2_3_3)
    pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')

    inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation=activation, name='inception_3a_1_1')
    inception_3a_3_3_reduce = conv_2d(pool2_3_3, 128, filter_size=1, activation=activation, name='inception_3a_3_3_reduce')
    inception_3a_3_3_asym_1 = conv_2d(inception_3a_3_3_reduce, 192, filter_size=[1,7],  activation=activation,name='inception_3a_3_3_asym_1')
    inception_3a_3_3 = conv_2d(inception_3a_3_3_asym_1, 192, filter_size=[7,1],  activation=activation,name='inception_3a_3_3')


    inception_3a_5_5_reduce = conv_2d(pool2_3_3, 32, filter_size=1, activation=activation, name = 'inception_3a_5_5_reduce')
    inception_3a_5_5_asym_1 = conv_2d(inception_3a_5_5_reduce, 96, filter_size=[1,7],  name = 'inception_3a_5_5_asym_1')
    inception_3a_5_5_asym_2 = conv_2d(inception_3a_5_5_asym_1, 96, filter_size=[7,1],  name = 'inception_3a_5_5_asym_2')
    inception_3a_5_5_asym_3 = conv_2d(inception_3a_5_5_asym_2, 96, filter_size=[1,7],  name = 'inception_3a_5_5_asym_3')
    inception_3a_5_5 = conv_2d(inception_3a_5_5_asym_3, 96, filter_size=[7,1],  name = 'inception_3a_5_5')

    
    inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
    inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation=activation, name='inception_3a_pool_1_1')

    # merge the inception_3a__
    inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

    inception_3b_1_1 = conv_2d(inception_3a_output, 128,filter_size=1,activation=activation, name= 'inception_3b_1_1' )


    inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation=activation, name='inception_3b_3_3_reduce')
    inception_3b_3_3_asym_1 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=[1,7],  activation=activation,name='inception_3b_3_3_asym_1')
    inception_3b_3_3 = conv_2d(inception_3b_3_3_asym_1, 192, filter_size=[7,1],  activation=activation,name='inception_3b_3_3')


    inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation=activation, name = 'inception_3b_5_5_reduce')
    inception_3b_5_5_asym_1 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=[1,7],  name = 'inception_3b_5_5_asym_1')
    inception_3b_5_5_asym_2 = conv_2d(inception_3b_5_5_asym_1, 96, filter_size=[7,1],  name = 'inception_3b_5_5_asym_2')
    inception_3b_5_5_asym_3 = conv_2d(inception_3b_5_5_asym_2, 96, filter_size=[1,7],  name = 'inception_3b_5_5_asym_3')
    inception_3b_5_5 = conv_2d(inception_3b_5_5_asym_3, 96, filter_size=[7,1],  name = 'inception_3b_5_5')





    inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
    inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1,activation=activation, name='inception_3b_pool_1_1')

    #merge the inception_3b_*
    inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat',axis=3,name='inception_3b_output')

    pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')
    inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation=activation, name='inception_4a_1_1')

    inception_4a_3_3_reduce = conv_2d(pool3_3_3, 128, filter_size=1, activation=activation, name='inception_4a_3_3_reduce')
    inception_4a_3_3_asym_1 = conv_2d(inception_4a_3_3_reduce, 192, filter_size=[1,7],  activation=activation,name='inception_4a_3_3_asym_1')
    inception_4a_3_3 = conv_2d(inception_4a_3_3_asym_1, 192, filter_size=[7,1],  activation=activation,name='inception_4a_3_3')


    inception_4a_5_5_reduce = conv_2d(pool3_3_3, 32, filter_size=1, activation=activation, name = 'inception_4a_5_5_reduce')
    inception_4a_5_5_asym_1 = conv_2d(inception_4a_5_5_reduce, 96, filter_size=[1,7],  name = 'inception_4a_5_5_asym_1')
    inception_4a_5_5_asym_2 = conv_2d(inception_4a_5_5_asym_1, 96, filter_size=[7,1],  name = 'inception_4a_5_5_asym_2')
    inception_4a_5_5_asym_3 = conv_2d(inception_4a_5_5_asym_2, 96, filter_size=[1,7],  name = 'inception_4a_5_5_asym_3')
    inception_4a_5_5 = conv_2d(inception_4a_5_5_asym_3, 96, filter_size=[7,1],  name = 'inception_4a_5_5')


    inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
    inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation=activation, name='inception_4a_pool_1_1')

    inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')


    inception_4b_1_1 = conv_2d(inception_4a_output, 160, filter_size=1, activation=activation, name='inception_4a_1_1')



    inception_4b_3_3_reduce = conv_2d(inception_4a_output, 128, filter_size=1, activation=activation, name='inception_4b_3_3_reduce')
    inception_4b_3_3_asym_1 = conv_2d(inception_4b_3_3_reduce, 192, filter_size=[1,7],  activation=activation,name='inception_4b_3_3_asym_1')
    inception_4b_3_3 = conv_2d(inception_4b_3_3_asym_1, 192, filter_size=[7,1],  activation=activation,name='inception_4b_3_3')


    inception_4b_5_5_reduce = conv_2d(inception_4a_output, 32, filter_size=1, activation=activation, name = 'inception_4b_5_5_reduce')
    inception_4b_5_5_asym_1 = conv_2d(inception_4b_5_5_reduce, 96, filter_size=[1,7],  name = 'inception_4b_5_5_asym_1')
    inception_4b_5_5_asym_2 = conv_2d(inception_4b_5_5_asym_1, 96, filter_size=[7,1],  name = 'inception_4b_5_5_asym_2')
    inception_4b_5_5_asym_3 = conv_2d(inception_4b_5_5_asym_2, 96, filter_size=[1,7],  name = 'inception_4b_5_5_asym_3')
    inception_4b_5_5 = conv_2d(inception_4b_5_5_asym_3, 96, filter_size=[7,1],  name = 'inception_4b_5_5')



    inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1,  name='inception_4b_pool')
    inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, filter_size=1, activation=activation, name='inception_4b_pool_1_1')

    inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], mode='concat', axis=3, name='inception_4b_output')


    inception_4c_1_1 = conv_2d(inception_4b_output, 128, filter_size=1, activation=activation,name='inception_4c_1_1')


    inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, filter_size=1, activation=activation, name='inception_4c_3_3_reduce')
    inception_4c_3_3_asym_1 = conv_2d(inception_4c_3_3_reduce, 192, filter_size=[1,7],  activation=activation,name='inception_4c_3_3_asym_1')
    inception_4c_3_3 = conv_2d(inception_4c_3_3_asym_1, 192, filter_size=[7,1],  activation=activation,name='inception_4c_3_3')


    inception_4c_5_5_reduce = conv_2d(inception_4b_output, 32, filter_size=1, activation=activation, name = 'inception_4c_5_5_reduce')
    inception_4c_5_5_asym_1 = conv_2d(inception_4c_5_5_reduce, 96, filter_size=[1,7],  name = 'inception_4c_5_5_asym_1')
    inception_4c_5_5_asym_2 = conv_2d(inception_4c_5_5_asym_1, 96, filter_size=[7,1],  name = 'inception_4c_5_5_asym_2')
    inception_4c_5_5_asym_3 = conv_2d(inception_4c_5_5_asym_2, 96, filter_size=[1,7],  name = 'inception_4c_5_5_asym_3')
    inception_4c_5_5 = conv_2d(inception_4c_5_5_asym_3, 96, filter_size=[7,1],  name = 'inception_4c_5_5')

    inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
    inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, filter_size=1, activation=activation, name='inception_4c_pool_1_1')

    inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], mode='concat', axis=3,name='inception_4c_output')


    
    pool5_7_7 = avg_pool_2d(inception_4c_output, kernel_size=7, strides=1)
    pool5_7_7 = dropout(pool5_7_7,dropout_value)
    loss = fully_connected(pool5_7_7, 2,activation='softmax')
    if(training):
        network = regression(loss, optimizer=optimizer,
                            loss='categorical_crossentropy',
                            learning_rate=0.001)
    else:
        network = loss;

    model = tflearn.DNN(network, checkpoint_path='inceptionv2_b_5',
                        max_checkpoints=1, tensorboard_verbose=0)
    return model


