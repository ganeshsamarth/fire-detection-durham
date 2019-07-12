'''
Autokeras implementation
'''

import autokeras as ak
import h5py
output_directory='/home/pbu/Desktop/ganesh-samarth-fire-detection-copy'
MODEL_DIR='/home/pbu/fire-detection-cnn/models_best/autokeras_model_SP.h5'
h5f = h5py.File("{}data_superpixel_test_centre.h5".format(output_directory), 'r')
final_test_accuracy=list()
final_f_score=list()
final_precision=list()
final_tpr=list()
final_fnr=list()
test_X = h5f['X']
test_Y = h5f['Y']
model=ak.ImageClassifier(verbose=True)

h5f1 = h5py.File("{}data_superpixel_with_red_centre_nonaugmented.h5".format(output_directory), 'r')
train_X = h5f1['X']
train_Y = h5f1['Y']
print(train_X.shape)
#print(train_Y.shape)
model.fit(train_X,train_Y[:,0],time_limit=60*60*6)
model.export_autokeras_model(MODEL_DIR)
#model.export_keras_model(MODEL_DIR)
#model.load_searcher().load_best_model().produce_keras_model().save(MODEL_DIR)


model.final_fit(train_X,train_Y[:,0],test_X,test_Y[:,0],retrain=True)
#model.export_keras_model(MODEL_DIR)
score=model.evaluate(test_X,test_Y[:,0])
predictions=model.predict(test_X)
print(score)
print(predictions)


