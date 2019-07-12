

import sys
sys.path.insert(0, '/home/pbu/fire-detection-durham/keras_models')
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
import h5py
import os
import sys

################################################################################


def evaluate_in_train_keras(model_name,model,num_epoch):
	output_directory='/home/pbu/Desktop/ganesh-samarth-fire-detection-copy'


	h5f = h5py.File("{}data_test.h5".format(output_directory), 'r')
	final_test_accuracy=list()
	final_f_score=list()	
	final_precision=list()
	final_tpr=list()
	final_fnr=list()
	test_X = h5f['X']
	test_Y = h5f['Y']
	input_shape=[224,224,3]
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


	if num_epoch==0:
		acc_thresh=sum(final_test_accuracy)/len(final_test_accuracy)
		with open('accuracy_thresh.txt','w') as myfile:
			myfile.write(str(acc_thresh)+':'+sys.argv[1]+'_keras.h5'+':'+sys.argv[1])
			model.save('models_weights_with_eval/'+sys.argv[1]+'/'+sys.argv[1]+'_keras.h5')
			string_check=sys.argv[1]+'_'+'keras.h5'
			
			for item in os.listdir('models_weights_with_eval/'+sys.argv[1]):
				
				if string_check in item:
					
					if os.path.isdir('/home/pbu/fire-detection-cnn/models_best/'+sys.argv[1])==False:
						os.mkdir('/home/pbu/fire-detection-cnn/models_best/'+sys.argv[1])

					
					os.system('cp /home/pbu/fire-detection-cnn/models_weights_with_eval/'+sys.argv[1]+'/'+item+' '+'/home/pbu/fire-detection-cnn/models_best/'+sys.argv[1]+'/')
				
			
			myfile.close()
	
	else:

		with open('accuracy_thresh.txt','r') as myfile2:
				data=myfile2.read()
				print(data.split(':'))
				acc_thresh=float(data.split(':')[0])
				save_path=data.split(':')[1]
				model_name=data.split(':')[2]
				myfile2.close()
			
		if sum(final_test_accuracy)/len(final_test_accuracy)>acc_thresh:
			acc_thresh=sum(final_test_accuracy)/len(final_test_accuracy)
			best_model=model
			print(save_path)
			with open('accuracy_thresh.txt','w') as myfile3:
				myfile3.write(str(acc_thresh)+':'+save_path+':'+model_name)
			myfile3.close()	
			best_model.save('models_weights_with_eval/'+model_name+'/'+save_path)
			print('saved model after epoch number:'+str(num_epoch))
			string_check=save_path
			for item in os.listdir('models_weights_with_eval/'+model_name):
				if string_check in item:
					os.system('cp /home/pbu/fire-detection-cnn/models_weights_with_eval/'+model_name+'/'+item+' '+'/home/pbu/fire-detection-cnn/models_best/'+model_name+'/')
			
			


	
	
	print('Test accuracy: '+ str(sum(final_test_accuracy)/len(final_test_accuracy)))
	print('Test f1_score: '+str(sum(final_f_score)/len(final_f_score)))
	print('Test precision_score: '+str(sum(final_precision)/len(final_precision)))
	print('Test tpr: '+ str(sum(final_tpr)/len(final_tpr)))
	print('Test fnr: '+str(sum(final_fnr)/len(final_fnr)))


	h5f.close()
	

	

	
