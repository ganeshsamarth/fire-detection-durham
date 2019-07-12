

import sys
sys.path.insert(0, '/home/pbu/fire-detection-durham/tflearn_models')

import tensorflow as tf
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



import h5py
import sys
import os
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score


################################################################################
'''
Superpixels have four kinds of data which has been tested,
best has proven to be centre nonaugmented
'''

################################################################################
import shutil
 
def copyDirectory(src, dest):
    try:
        shutil.copytree(src, dest)
    # Directories are the same
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print('Directory not copied. Error: %s' % e)


def evaluate_in_train_superpixel(model_name,model,num_epoch):
	output_directory='/home/pbu/Desktop/ganesh-samarth-fire-detection-copy'


	h5f = h5py.File("{}data_superpixel_test_centre.h5".format(output_directory), 'r')
	final_test_accuracy=list()
	final_f_score=list()
	final_precision=list()
	final_tpr=list()
	final_fnr=list()
	test_X = h5f['X']
	test_Y = h5f['Y']
	batch_size=100
	num_batches=len(test_X)//batch_size
	for i in range(0,num_batches):
		ypred=model.predict(test_X[i*batch_size:(i+1)*batch_size])
		print(ypred.shape)
		ypred=np.array(ypred)
		ypred_list=list(ypred[:,0])
		test_Y_act=list(test_Y[i*batch_size:(i+1)*batch_size,0])
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

		f_score=f1_score(list(test_Y[i*batch_size:(i+1)*batch_size,0]),ypred_act)
		pres=precision_score(list(test_Y[i*batch_size:(i+1)*batch_size,0]),ypred_act)
		
		final_f_score.append(f_score)
		final_precision.append(pres)
		final_tpr.append(tpr/p)
		final_fnr.append(fnr/f)
		final_test_accuracy.append((tpr+fnr)/(p+f))
	if num_epoch==0:
		acc_thresh=sum(final_test_accuracy)/len(final_test_accuracy)
		with open('accuracy_thresh.txt','w') as myfile:
			myfile.write(str(acc_thresh)+':'+sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3]+'_'+str(sys.argv[4])+'_'+sys.argv[5]+'_150epoch'+'.tflearn'+':'+sys.argv[1])
			model.save('models_new_weights_superpixels/centre_nonaugmented_with_red_with_transfer/'+sys.argv[1]+'/'+sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3]+'_'+str(sys.argv[4])+'_'+sys.argv[5]+'_150epoch'+'.tflearn')
			string_check=sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3]+'_'+str(sys.argv[4])+'_'+sys.argv[5]+'.tflearn'
			
			for item in os.listdir('models_new_weights_superpixels/centre_nonaugmented_with_red_with_transfer/'+sys.argv[1]):
				
				if string_check in item:
					
					if os.path.isdir('/home/pbu/fire-detection-cnn/models_best_superpixels/transfer/'+sys.argv[1])==False:
						os.mkdir('/home/pbu/fire-detection-cnn/models_best_superpixels/transfer/'+sys.argv[1])

					
					os.system('cp /home/pbu/fire-detection-cnn/models_new_weights_superpixels/centre_nonaugmented_with_red_with_transfer/'+sys.argv[1]+'/'+item+' '+'/home/pbu/fire-detection-cnn/models_best_superpixels/transfer/'+sys.argv[1]+'/')
				
			
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
			best_model.save('models_new_weights_superpixels/centre_nonaugmented_with_red_with_transfer/'+model_name+'/'+save_path)
			print('saved model after epoch number:'+str(num_epoch))
			string_check=save_path
			for item in os.listdir('models_new_weights_superpixels/centre_nonaugmented_with_red_with_transfer/'+model_name):
				if string_check in item:
					os.system('cp /home/pbu/fire-detection-cnn/models_new_weights_superpixels/centre_nonaugmented_with_red_with_transfer/'+model_name+'/'+item+' '+'/home/pbu/fire-detection-cnn/models_best_superpixels/transfer/'+model_name+'/')
			
			


	
	
	print('Test accuracy: '+ str(sum(final_test_accuracy)/len(final_test_accuracy)))
	print('Test f1_score: '+str(sum(final_f_score)/len(final_f_score)))
	print('Test precision_score: '+str(sum(final_precision)/len(final_precision)))
	print('Test tpr: '+ str(sum(final_tpr)/len(final_tpr)))
	print('Test fnr: '+str(sum(final_fnr)/len(final_fnr)))


	h5f.close()
	

	

	
