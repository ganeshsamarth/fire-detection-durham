'''
Training data augmentation for superpixel classification
'''
import Augmentor

p=Augmentor.Pipeline('/home/capture/Downloads/fire-dataset-dunnings/superpixels/isolated-superpixels/test/fire')
p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)
p.sample(135000)