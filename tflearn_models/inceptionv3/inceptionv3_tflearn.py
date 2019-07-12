'''
InceptionV3
combination of A,B and C blocks
tensorflow inceptionv3 module with grid size reduction
'''
from __future__ import division, print_function, absolute_import

import tflearn
import h5py
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d,global_avg_pool
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression


def construct_inceptionv3(x,y,normalization_method,optimizer,dropout_value,training=True):
    # Build network
    # 224 x 224 original size
    network = input_data(shape=[None, y, x, 3])
    conv1_3_3 =conv_2d(network, 32, 3, strides=2, activation='relu', name = 'conv1_3_3',padding='valid')
    conv2_3_3 =conv_2d(conv1_3_3, 32, 3, strides=1, activation='relu', name = 'conv2_3_3',padding='valid')
    conv3_3_3 =conv_2d(conv2_3_3, 64, 3, strides=2, activation='relu', name = 'conv3_3_3')

    pool1_3_3 = max_pool_2d(conv3_3_3, 3,strides=2)
    pool1_3_3 = globals()[normalization_method](pool1_3_3)
    conv1_7_7 =conv_2d(pool1_3_3, 80,3, strides=1, activation='relu', name='conv2_7_7_s2',padding='valid')
    conv2_7_7 =conv_2d(conv1_7_7, 192,3, strides=1, activation='relu', name='conv2_7_7_s2',padding='valid')
    pool2_3_3=max_pool_2d(conv2_7_7,3,strides=2)

    inception_3a_1_1 = conv_2d(pool2_3_3,64, filter_size=1, activation='relu', name='inception_3a_1_1')

    inception_3a_3_3_reduce = conv_2d(pool2_3_3, 48, filter_size=1, activation='relu', name='inception_3a_3_3_reduce')
    inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 64, filter_size=[5,5],  activation='relu',name='inception_3a_3_3')


    inception_3a_5_5_reduce = conv_2d(pool2_3_3, 64, filter_size=1, activation='relu', name = 'inception_3a_5_5_reduce')
    inception_3a_5_5_asym_1 = conv_2d(inception_3a_5_5_reduce, 96, filter_size=[3,3],  name = 'inception_3a_5_5_asym_1')
    inception_3a_5_5 = conv_2d(inception_3a_5_5_asym_1, 96, filter_size=[3,3],  name = 'inception_3a_5_5')


    inception_3a_pool = avg_pool_2d(pool2_3_3, kernel_size=3, strides=1,  name='inception_3a_pool')
    inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')

    inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3, name='inception_3a_output')


    inception_3b_1_1 = conv_2d(inception_3a_output, 64, filter_size=1, activation='relu', name='inception_3a_1_1')



    inception_3b_3_3_reduce = conv_2d(inception_3a_output, 48, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
    inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 64, filter_size=[5,5],  activation='relu',name='inception_3b_3_3')


    inception_3b_5_5_reduce = conv_2d(inception_3a_output, 64, filter_size=1, activation='relu', name = 'inception_3b_5_5_reduce')
    inception_3b_5_5_asym_1 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=[3,3],  name = 'inception_3b_5_5_asym_1')
    inception_3b_5_5 = conv_2d(inception_3b_5_5_asym_1, 96, filter_size=[3,3],  name = 'inception_3b_5_5')



    inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
    inception_3b_pool_1_1 = conv_2d(inception_3b_pool,64 , filter_size=1, activation='relu', name='inception_3b_pool_1_1')

    inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat', axis=3, name='inception_3b_output')


    inception_3c_1_1 = conv_2d(inception_3b_output, 64, filter_size=1, activation='relu',name='inception_3c_1_1')

    inception_3c_3_3_reduce = conv_2d(inception_3b_output, 48, filter_size=1, activation='relu', name='inception_3c_3_3_reduce')
    inception_3c_3_3 = conv_2d(inception_3c_3_3_reduce, 64, filter_size=[5,5],  activation='relu',name='inception_3c_3_3')

    inception_3c_5_5_reduce = conv_2d(inception_3b_output, 64, filter_size=1, activation='relu', name = 'inception_3c_5_5_reduce')
    inception_3c_5_5_asym_1 = conv_2d(inception_3c_5_5_reduce, 96, filter_size=[3,3],  name = 'inception_3c_5_5_asym_1')
    inception_3c_5_5 = conv_2d(inception_3c_5_5_asym_1, 96, filter_size=[3,3],  name = 'inception_3c_5_5')

    inception_3c_pool = max_pool_2d(inception_3b_output, kernel_size=3, strides=1)
    inception_3c_pool_1_1 = conv_2d(inception_3c_pool, 64, filter_size=1, activation='relu', name='inception_3c_pool_1_1')

    inception_3c_output = merge([inception_3c_1_1, inception_3c_3_3, inception_3c_5_5, inception_3c_pool_1_1], mode='concat', axis=3,name='inception_3c_output')



    inception_4a_1_1 = conv_2d(inception_3c_output, 384, filter_size=3,strides=2, activation='relu',name='inception_4a_1_1',padding='valid')

    inception_4a_5_5_reduce = conv_2d(inception_3c_output, 64, filter_size=1, activation='relu', name = 'inception_4a_5_5_reduce')
    inception_4a_5_5_asym_1 = conv_2d(inception_4a_5_5_reduce, 96, filter_size=[3,3],  name = 'inception_4a_5_5_asym_1')
    inception_4a_5_5 = conv_2d(inception_4a_5_5_asym_1, 96,strides=2, filter_size=[3,3],  name = 'inception_4a_5_5',padding='valid')

    inception_4a_pool = max_pool_2d(inception_3c_output, kernel_size=3, strides=2,padding='valid')

    inception_4a_output = merge([inception_4a_1_1, inception_4a_5_5, inception_4a_pool], mode='concat', axis=3,name='inception_4a_output')

    inception_5a_1_1 = conv_2d(inception_4a_output, 192, 1, activation='relu', name='inception_5a_1_1')

    inception_5a_3_3_reduce = conv_2d(inception_4a_output, 128, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
    inception_5a_3_3_asym_1 = conv_2d(inception_5a_3_3_reduce, 128, filter_size=[1,7],  activation='relu',name='inception_5a_3_3_asym_1')
    inception_5a_3_3 = conv_2d(inception_5a_3_3_asym_1, 192, filter_size=[7,1],  activation='relu',name='inception_5a_3_3')


    inception_5a_5_5_reduce = conv_2d(inception_4a_output, 128, filter_size=1, activation='relu', name = 'inception_5a_5_5_reduce')
    inception_5a_5_5_asym_1 = conv_2d(inception_5a_5_5_reduce, 128, filter_size=[7,1],  name = 'inception_5a_5_5_asym_1')
    inception_5a_5_5_asym_2 = conv_2d(inception_5a_5_5_asym_1, 128, filter_size=[1,7],  name = 'inception_5a_5_5_asym_2')
    inception_5a_5_5_asym_3 = conv_2d(inception_5a_5_5_asym_2, 128, filter_size=[7,1],  name = 'inception_5a_5_5_asym_3')
    inception_5a_5_5 = conv_2d(inception_5a_5_5_asym_3, 192, filter_size=[1,7],  name = 'inception_5a_5_5')


    inception_5a_pool = avg_pool_2d(inception_4a_output, kernel_size=3, strides=1 )
    inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 192, filter_size=1, activation='relu', name='inception_5a_pool_1_1')

    # merge the inception_5a__
    inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], mode='concat', axis=3)


    inception_5b_1_1 = conv_2d(inception_5a_output, 192, 1, activation='relu', name='inception_5b_1_1')

    inception_5b_3_3_reduce = conv_2d(inception_5a_output, 160, filter_size=1, activation='relu', name='inception_5b_3_3_reduce')
    inception_5b_3_3_asym_1 = conv_2d(inception_5b_3_3_reduce, 160, filter_size=[1,7],  activation='relu',name='inception_5b_3_3_asym_1')
    inception_5b_3_3 = conv_2d(inception_5b_3_3_asym_1, 192, filter_size=[7,1],  activation='relu',name='inception_5b_3_3')


    inception_5b_5_5_reduce = conv_2d(inception_5a_output, 160, filter_size=1, activation='relu', name = 'inception_5b_5_5_reduce')
    inception_5b_5_5_asym_1 = conv_2d(inception_5b_5_5_reduce, 160, filter_size=[7,1],  name = 'inception_5b_5_5_asym_1')
    inception_5b_5_5_asym_2 = conv_2d(inception_5b_5_5_asym_1, 160, filter_size=[1,7],  name = 'inception_5b_5_5_asym_2')
    inception_5b_5_5_asym_3 = conv_2d(inception_5b_5_5_asym_2, 160, filter_size=[7,1],  name = 'inception_5b_5_5_asym_3')
    inception_5b_5_5 = conv_2d(inception_5b_5_5_asym_3, 192, filter_size=[1,7],  name = 'inception_5b_5_5')


    inception_5b_pool = avg_pool_2d(inception_5a_output, kernel_size=3, strides=1 )
    inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 192, filter_size=1, activation='relu', name='inception_5b_pool_1_1')

    # merge the inception_5b__
    inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], mode='concat', axis=3)

    inception_5c_1_1 = conv_2d(inception_5b_output, 192, 1, activation='relu', name='inception_5c_1_1')

    inception_5c_3_3_reduce = conv_2d(inception_5b_output, 160, filter_size=1, activation='relu', name='inception_5c_3_3_reduce')
    inception_5c_3_3_asym_1 = conv_2d(inception_5c_3_3_reduce, 160, filter_size=[1,7],  activation='relu',name='inception_5c_3_3_asym_1')
    inception_5c_3_3 = conv_2d(inception_5c_3_3_asym_1, 192, filter_size=[7,1],  activation='relu',name='inception_5c_3_3')


    inception_5c_5_5_reduce = conv_2d(inception_5b_output, 160, filter_size=1, activation='relu', name = 'inception_5c_5_5_reduce')
    inception_5c_5_5_asym_1 = conv_2d(inception_5c_5_5_reduce, 160, filter_size=[7,1],  name = 'inception_5c_5_5_asym_1')
    inception_5c_5_5_asym_2 = conv_2d(inception_5c_5_5_asym_1, 160, filter_size=[1,7],  name = 'inception_5c_5_5_asym_2')
    inception_5c_5_5_asym_3 = conv_2d(inception_5c_5_5_asym_2, 160, filter_size=[7,1],  name = 'inception_5c_5_5_asym_3')
    inception_5c_5_5 = conv_2d(inception_5c_5_5_asym_3, 192, filter_size=[1,7],  name = 'inception_5c_5_5')


    inception_5c_pool = avg_pool_2d(inception_5b_output, kernel_size=3, strides=1 )
    inception_5c_pool_1_1 = conv_2d(inception_5c_pool, 192, filter_size=1, activation='relu', name='inception_5c_pool_1_1')

    # merge the inception_5c__
    inception_5c_output = merge([inception_5c_1_1, inception_5c_3_3, inception_5c_5_5, inception_5c_pool_1_1], mode='concat', axis=3)




    inception_5d_1_1 = conv_2d(inception_5c_output, 192, 1, activation='relu', name='inception_5d_1_1')

    inception_5d_3_3_reduce = conv_2d(inception_5c_output, 192, filter_size=1, activation='relu', name='inception_5d_3_3_reduce')
    inception_5d_3_3_asym_1 = conv_2d(inception_5d_3_3_reduce, 192, filter_size=[1,7],  activation='relu',name='inception_5d_3_3_asym_1')
    inception_5d_3_3 = conv_2d(inception_5d_3_3_asym_1, 192, filter_size=[7,1],  activation='relu',name='inception_5d_3_3')


    inception_5d_5_5_reduce = conv_2d(inception_5c_output, 192, filter_size=1, activation='relu', name = 'inception_5d_5_5_reduce')
    inception_5d_5_5_asym_1 = conv_2d(inception_5d_5_5_reduce, 192, filter_size=[7,1],  name = 'inception_5d_5_5_asym_1')
    inception_5d_5_5_asym_2 = conv_2d(inception_5d_5_5_asym_1, 192, filter_size=[1,7],  name = 'inception_5d_5_5_asym_2')
    inception_5d_5_5_asym_3 = conv_2d(inception_5d_5_5_asym_2, 192, filter_size=[7,1],  name = 'inception_5d_5_5_asym_3')
    inception_5d_5_5 = conv_2d(inception_5d_5_5_asym_3, 192, filter_size=[1,7],  name = 'inception_5d_5_5')


    inception_5d_pool = avg_pool_2d(inception_5c_output, kernel_size=3, strides=1 )
    inception_5d_pool_1_1 = conv_2d(inception_5d_pool, 192, filter_size=1, activation='relu', name='inception_5d_pool_1_1')

    # merge the inception_5d__
    inception_5d_output = merge([inception_5d_1_1, inception_5d_3_3, inception_5d_5_5, inception_5d_pool_1_1], mode='concat', axis=3)


    inception_6a_1_1=conv_2d(inception_5d_output,192,1,activation='relu',name='inception_6a_1_1')
    inception_6a_1_3=conv_2d(inception_6a_1_1,320,3,strides=2,padding='valid',name='inception_6a_1_3')

    inception_6a_3_3_reduce = conv_2d(inception_5d_output, 192, filter_size=1, activation='relu', name='inception_6a_3_3_reduce')
    inception_6a_3_3_asym_1 = conv_2d(inception_6a_3_3_reduce, 192, filter_size=[1,7],  activation='relu',name='inception_6a_3_3_asym_1')
    inception_6a_3_3_asym_2 = conv_2d(inception_6a_3_3_asym_1, 192, filter_size=[7,1],  activation='relu',name='inception_6a_3_3_asym_2')
    inception_6a_3_3=conv_2d(inception_6a_3_3_asym_2,192,3,strides=2,activation='relu',padding='valid',name='inception_6a_3_3')

    inception_6a_pool=max_pool_2d(inception_5d_output,kernel_size=3,strides=2,padding='valid')

    inception_6a_output=merge([inception_6a_1_3,inception_6a_3_3,inception_6a_pool],mode='concat',axis=3)


    inception_7a_1_1 = conv_2d(inception_6a_output, 320, 1, activation='relu', name='inception_7a_1_1')
    inception_7a_3_3_reduce = conv_2d(inception_6a_output, 384, filter_size=1, activation='relu', name='inception_7a_3_3_reduce')
    inception_7a_3_3_asym_1 = conv_2d(inception_7a_3_3_reduce, 384, filter_size=[1,3],  activation='relu',name='inception_7a_3_3_asym_1')
    inception_7a_3_3_asym_2 = conv_2d(inception_7a_3_3_reduce, 384, filter_size=[3,1],  activation='relu',name='inception_7a_3_3_asym_2')
    inception_7a_3_3=merge([inception_7a_3_3_asym_1,inception_7a_3_3_asym_2],mode='concat',axis=3)

    inception_7a_5_5_reduce = conv_2d(inception_6a_output, 448, filter_size=1, activation='relu', name = 'inception_7a_5_5_reduce')
    inception_7a_5_5_asym_1 = conv_2d(inception_7a_5_5_reduce, 384, filter_size=[3,3],  name = 'inception_7a_5_5_asym_1')
    inception_7a_5_5_asym_2 = conv_2d(inception_7a_3_3_asym_1, 384, filter_size=[1,3],  activation='relu',name='inception_7a_5_5_asym_2')
    inception_7a_5_5_asym_3 = conv_2d(inception_7a_3_3_asym_1, 384, filter_size=[3,1],  activation='relu',name='inception_7a_5_5_asym_3')
    inception_7a_5_5=merge([inception_7a_5_5_asym_2,inception_7a_5_5_asym_3],mode='concat',axis=3)


    inception_7a_pool = avg_pool_2d(inception_6a_output, kernel_size=3, strides=1 )
    inception_7a_pool_1_1 = conv_2d(inception_7a_pool, 192, filter_size=1, activation='relu', name='inception_7a_pool_1_1')

    # merge the inception_7a__
    inception_7a_output = merge([inception_7a_1_1, inception_7a_3_3, inception_7a_5_5, inception_7a_pool_1_1], mode='concat', axis=3)

    inception_7b_1_1 = conv_2d(inception_7a_output, 320, 1, activation='relu', name='inception_7b_1_1')
    inception_7b_3_3_reduce = conv_2d(inception_7a_output, 384, filter_size=1, activation='relu', name='inception_7b_3_3_reduce')
    inception_7b_3_3_asym_1 = conv_2d(inception_7b_3_3_reduce, 384, filter_size=[1,3],  activation='relu',name='inception_7b_3_3_asym_1')
    inception_7b_3_3_asym_2 = conv_2d(inception_7b_3_3_reduce, 384, filter_size=[3,1],  activation='relu',name='inception_7b_3_3_asym_2')
    inception_7b_3_3=merge([inception_7b_3_3_asym_1,inception_7b_3_3_asym_2],mode='concat',axis=3)

    inception_7b_5_5_reduce = conv_2d(inception_7a_output, 448, filter_size=1, activation='relu', name = 'inception_7b_5_5_reduce')
    inception_7b_5_5_asym_1 = conv_2d(inception_7b_5_5_reduce, 384, filter_size=[3,3],  name = 'inception_7b_5_5_asym_1')
    inception_7b_5_5_asym_2 = conv_2d(inception_7b_3_3_asym_1, 384, filter_size=[1,3],  activation='relu',name='inception_7b_5_5_asym_2')
    inception_7b_5_5_asym_3 = conv_2d(inception_7b_3_3_asym_1, 384, filter_size=[3,1],  activation='relu',name='inception_7b_5_5_asym_3')
    inception_7b_5_5=merge([inception_7b_5_5_asym_2,inception_7b_5_5_asym_3],mode='concat',axis=3)


    inception_7b_pool = avg_pool_2d(inception_7a_output, kernel_size=3, strides=1 )
    inception_7b_pool_1_1 = conv_2d(inception_7b_pool, 192, filter_size=1, activation='relu', name='inception_7b_pool_1_1')

    # merge the inception_7b__
    inception_7b_output = merge([inception_7b_1_1, inception_7b_3_3, inception_7b_5_5, inception_7b_pool_1_1], mode='concat', axis=3)


    pool5_7_7=global_avg_pool(inception_7b_output)
    pool5_7_7=dropout(pool5_7_7,dropout_value)
    loss = fully_connected(pool5_7_7, 2,activation='softmax')

    if(training):
        network = regression(loss, optimizer=optimizer,
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
    else:
        network=loss

    model = tflearn.DNN(network, checkpoint_path='inceptionv3',
                        max_checkpoints=1, tensorboard_verbose=0)

    return model
