

import sys
sys.path.insert(0, '/home/pbu/fire-detection-durham/tflearn_models')
from inceptionv2.inceptionv2_tflearn_a import *
from inceptionv2.inceptionv2_tflearn_b import *
from inceptionv2.inceptionv2_tflearn_c import *
from inceptionv2.inceptionv2_tflearn_a_4 import *
from inceptionv2.inceptionv2_tflearn_a_5 import *
from inceptionv2.inceptionv2_tflearn_a_6 import *
from inceptionv2.inceptionv2_tflearn_b_4 import *
from inceptionv2.inceptionv2_tflearn_b_5 import *
from inceptionv2.inceptionv2_tflearn_b_6 import *
from inceptionv2.inceptionv2_tflearn_c_4 import *
from inceptionv2.inceptionv2_tflearn_c_5 import *
from inceptionv2.inceptionv2_tflearn_c_6 import *
from inceptionv2.tflearn_inceptionv2onfire_a import *
from inceptionv2.tflearn_inceptionv2onfire_b import *
from inceptionv2.tflearn_inceptionv2onfire_c import *
from inceptionv3.inceptionv3_a_tflearn import *
from inceptionv3.inceptionv3_b_tflearn import *
from inceptionv3.inceptionv3_c_tflearn import *
from inceptionv3.inceptionv3_d_tflearn import *
from inceptionv3.inceptionv3_e_tflearn import *
from inceptionv3.inceptionv3_f_tflearn import *
from inceptionv3.inceptionv3_g_tflearn import *
from inceptionv3.inceptionv3_h_tflearn import *
from inceptionv3.inceptionv3_i_tflearn import *
from inceptionv3.inceptionv3_j_tflearn import *
from inceptionv3.inceptionv3_k_tflearn import *
from inceptionv3.inceptionv3_l_tflearn import *
from inceptionv4.inceptionv4_tflearn import *
from inceptionv4.inceptionv4_a_tflearn import *
from inceptionv4.inceptionv4_b_tflearn import *
from inceptionv4.inceptionv4_c_tflearn import *
from inceptionv4.inceptionv4_d_tflearn import *
from inceptionv4.inceptionv4_e_tflearn import *
from inceptionv4.inceptionv4_f_tflearn import *
from inceptionv4.inceptionv4_g_tflearn import *
from inceptionv4.inceptionv4_h_tflearn import *
from inceptionv4.inceptionv4_i_tflearn import *
from inceptionv4.inceptionv4_j_tflearn import *
from inceptionv4.inceptionv4_k_tflearn import *
from inceptionv4.inceptionv4_l_tflearn import *
from inceptionv4.inceptionv4_m_tflearn import *
import h5py
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score


################################################################################



################################################################################

   #   construct and display model
'''
    model = construct_inceptionv1onfire (224, 224, training=False)
    print("Constructed InceptionV1-OnFire ...")

    model.load(os.path.join("models/SP-InceptionV1-OnFire", "sp-inceptiononv1onfire"),weights_only=True)
    print("Loaded CNN network weights ...")
'''
################################################################################

model_name=globals()['construct_'+sys.argv[1]]

output_directory='/home/pbu/Desktop/ganesh-samarth-fire-detection-copy'
num_epochs=15

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



model = model_name(224, 224,normalization,optimizer,dropout,activation,training=False)
print("Constructed ..."+sys.argv[1])
model.load("/home/pbu/fire-detection-cnn/models_best/"+sys.argv[1]+'/'+sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3]+'_'+str(sys.argv[4])+'_'+sys.argv[5]+'.tflearn')

h5f = h5py.File("{}data_test.h5".format(output_directory), 'r')
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



print('Test accuracy: '+ str(sum(final_test_accuracy)/len(final_test_accuracy)))
print('Test f1_score: '+str(sum(final_f_score)/len(final_f_score)))
print('Test precision_score: '+str(sum(final_precision)/len(final_precision)))
print('Test tpr: '+ str(sum(final_tpr)/len(final_tpr)))
print('Test fnr: '+str(sum(final_fnr)/len(final_fnr)))
#print(p)
#print(f)

with open('test_results_new.txt','a') as myfile:
	myfile.write('models_new_weights/'+sys.argv[1]+'/'+sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3]+'_'+str(sys.argv[4])+'_test'+'.tflearn'+'\n')
	myfile.write('Test accuracy: '+ str(sum(final_test_accuracy)/len(final_test_accuracy))+'\n')
	myfile.write('Test f1_score: '+str(sum(final_f_score)/len(final_f_score))+'\n')
	myfile.write('Test precision_score: '+str(sum(final_precision)/len(final_precision))+'\n')
	myfile.write('Test tpr: '+ str(sum(final_tpr)/len(final_tpr))+'\n')
	myfile.write('Test fnr: '+str(sum(final_fnr)/len(final_fnr))+'\n')

h5f.close()

