'''
Tflearn train models for binary fire detection
'''


import tensorflow as tf
from evaluate_in_train_tflearn import *







model_name=globals()['construct_'+sys.argv[1]]

output_directory='/home/pbu/Desktop/ganesh-samarth-fire-detection-copy'

if sys.argv[2]=='lrn':
	normalization='local_response_normalization'
else:
	normalization='batch_normalization'


if sys.argv[3]=='rmsprop':
	optimizer='rmsprop'
	num_epochs=20
else:
	optimizer='momentum'



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




h5f1 = h5py.File("{}data.h5".format(output_directory), 'r')
train_X = h5f1['X']
train_Y = h5f1['Y']
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

for i in range(0,30):
	
	model.fit(train_X,train_Y,n_epoch=1,validation_set=0.2,batch_size=64,show_metric=True,shuffle=True)
	#model.save('models_weights_with_eval/'+sys.argv[1]+'/'+sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3]+'_'+str(sys.argv[4])+str(i)+'.tflearn')
	evaluate_in_train(sys.argv[1],model,i)
	





with open('parameters_count.txt','a') as myfile2:
	myfile2.write('models_new_weights/'+sys.argv[1]+'/'+sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3]+'_'+str(sys.argv[4])+'.tflearn'+'\n')
	myfile2.write(str(total_parameters)+'\n')

h5f1.close()

