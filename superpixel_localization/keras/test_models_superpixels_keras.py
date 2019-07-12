'''

Testing keras models for superpixels

'''


from resnet_keras import *
import numpy as np
from efficientnet_keras import *
from inception_resnet_v1_keras import *
from inception_resnet_v2_keras import *
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
import keras
import h5py
import sys

model_name=sys.argv[1]
output_directory='/home/pbu/Desktop/ganesh-samarth-fire-detection-copy'

h5f = h5py.File("{}data_superpixel_test_centre.h5".format(output_directory), 'r')
final_test_accuracy=list()
final_f_score=list()
final_precision=list()
final_tpr=list()
final_fnr=list()
test_X = h5f['X']
test_Y = h5f['Y']
input_shape=[224,224,3]
model=globals()[model_name](2)
model.load_weights(model_name+"_keras.h5")
ypred=model.predict(test_X)
print(ypred.shape)
ypred=np.array(ypred)
ypred_list=list(ypred[:,0])
test_Y_act=list(test_Y[:,0])
ypred_act=list()
for k in ypred_list:
	if k>0.5:
		ypred_act.append(1)
	else:
		ypred_act.append(0)
tpr=0
p=0
f=0
fnr=0
for j in range (0,len(ypred_act)):
	if ypred_act[j]==1 and test_Y_act[j]==1:
		tpr+=1
		p+=1
	if ypred_act[j]==1 and test_Y_act[j]==0:
		f+=1
	if ypred_act[j]==0 and test_Y_act[j]==1:
		p+=1
	if ypred_act[j]==0 and test_Y_act[j]==0:
		fnr+=1
		f+=1

f_score=f1_score(list(test_Y[:,0]),ypred_act)
pres=precision_score(list(test_Y[:,0]),ypred_act)
	
final_f_score.append(f_score)
final_precision.append(pres)
final_tpr.append(tpr/p)
final_fnr.append(fnr/f)
final_test_accuracy.append((tpr+fnr)/(p+f))



print('Test accuracy: '+ str(sum(final_test_accuracy)/len(final_test_accuracy)))
print('Test f1_score: '+str(sum(final_f_score)/len(final_f_score)))
print('Test precision_score: '+str(sum(final_precision)/len(final_precision)))
print('Test tpr: '+ str(sum(final_tpr)/len(final_tpr)))
print('Test fnr: '+str(sum(final_fnr)/len(final_fnr)))
#print(p)
#print(f)

with open('test_results_new.txt','a') as myfile:
	myfile.write(model_name+'_keras'+'\n')
	myfile.write('Test accuracy: '+ str(sum(final_test_accuracy)/len(final_test_accuracy))+'\n')
	myfile.write('Test f1_score: '+str(sum(final_f_score)/len(final_f_score))+'\n')
	myfile.write('Test precision_score: '+str(sum(final_precision)/len(final_precision))+'\n')
	myfile.write('Test tpr: '+ str(sum(final_tpr)/len(final_tpr))+'\n')
	myfile.write('Test fnr: '+str(sum(final_fnr)/len(final_fnr))+'\n')
	myfile.write('\n')

h5f.close()
