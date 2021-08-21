#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:32:36 2020

This an supporting code for Similarity Differrence and Uniqness for Explainable AI (SIDU) paper.

This code will run the evalautions on single image and as well as for all the images in the given dataset.

# TO use this code install tensorflow, numpy, matplotlib,skimage, tqdm, PIL, os modules

@author: satya (email: smmu@create.aau.dk)
"""

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import os
from matplotlib import pyplot as plt
from skimage.transform import resize
from tqdm import tqdm
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions
from tensorflow.python.keras.models import load_model
from keras import backend as K
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow.keras import layers,models
import SIDU_XAI as XAI

#import grad_cam as gdc
import cv2
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os.path import join, exists
from PIL import ImageFile





if __name__ == '__main__':


    
 #imagelist = glob.glob('/home/administrator/DATA/ILSVRC2012/ILSVRC2012_img_val_1to1000/*.JPEG') 
 
 ## path of your base directory to run on whole dataset.
 
 base_dir = '/home/administrator/DATA/ILSVRC2012'
 
 ## CHOOSING THE base MODEL TO GET EXPLANTIONS
 model_eval = 'Resnet50'

## path of your Sub directory to run on whole dataset
 valid_data_dir = os.path.join(base_dir, 'validate_pub')

 test_datagen = ImageDataGenerator(rescale=1./255)
 test_generator = test_datagen.flow_from_directory(
    directory= valid_data_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False)
 
 ### to load the based model here we choose Resnet-50 and VGG-16 but we can any models
 if model_eval == 'Resnet50':
    base_model = ResNet50()
    features_model = Model(inputs=base_model.input, outputs=base_model.get_layer('activation_48').output)
    
 elif model_eval == 'Vgg16':
    base_model = VGG19(weights='imagenet')
    features_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
    
 ## to evalute on single image for different state-of-art methods
 evaluate = 'single_image'
 if evaluate == 'single_image':
     
     run = 'my_method'
     ## read the image path
     read_path = join('test_images', 'water-bird.JPEG')
     
     ## load the image 
     img, x = XAI.load_img(read_path,(224,224))
     img1, img_tensor1 = XAI.load_image(read_path, (224,224), show=True)

     if run == 'my_method':
           #read_path = imagelist[0]
           ImageFile.LOAD_TRUNCATED_IMAGES = True
           
           conv_features = features_model.predict(x)
           last_conv_output = np.squeeze(conv_features)
           ## genereating the feature image masks
           masks, grid, cell_size, up_size = XAI.generate_masks_conv_output((224,224), last_conv_output, 8)
           ## to plot and see the each inidiviadual mask
           new_masks = np.rollaxis(masks, 2, 0)
           size = new_masks.shape
           data = new_masks.reshape(size[0], size[1], size[2], 1)
           conv_out = last_conv_output[:,:,100]
           conv_100 = conv_out > 0.5
           conv_ot100 = conv_100.astype('float32')
           mask_ind = masks[:, :, 100]
           grid_ind = grid[100,:,:]
           new_mask= np.reshape(mask_ind,(224,224))
           masked = x * data
           masked_2 = img_tensor1 * data
           plt.subplot(1, 4, 3)
           plt.imshow(new_mask)
           plt.axis('off')
           plt.subplot(1, 4, 1)
           plt.imshow(grid_ind)
           plt.axis('off')
           plt.subplot(1, 4, 4)
           plt.imshow(masked_2[100,:,:])
           plt.axis('off')
           plt.subplot(1, 4, 2)
           plt.imshow(conv_ot100)
           plt.axis('off')
           # #masks, grid, cell_size, up_size = XAI.generate_masks_conv_output((512,512), conv_features, 8)
         
           N = len(data)
           save_path = XAI.fold_dir('./IMAGENET_SIDU_HEATMAPS_single_image_save')
           ## choose the different versions of my method 
           my_method = 'SIDU'
          
          
           ### with sim diff_and_uniqness
           if my_method == 'SIDU':
               sal, weights, new_interactions, diff_interactions, pred_org = XAI.explain_SIDU(base_model, x, N, 0.5, data, (224,224))
               pred = np.argmax(pred_org)
               class_idx = pred
               dst_image = os.path.splitext(read_path.split('/')[-1])[0]+'_SIDU_XAI'+'.jpg'
               XAI._show_explanation(img, sal[class_idx], class_idx,  read_path+'imagenet', show=True, save=True, save_folder=save_path, alpha=0.3, figsize=(15, 5),
                        cmap='jet')
               sal_exp = sal[pred]
               cv2.normalize(sal_exp, sal_exp, 0, 255, cv2.NORM_MINMAX)
               dst_path = os.path.join(save_path, dst_image)
               cv2.imwrite(dst_path, sal_exp) 
           
           dst_plot = os.path.splitext(read_path.split('/')[-1])[0]+'_my_method_SIDU_plot'+'.png'

           dst_path2 = os.path.join(save_path, dst_plot)      
           plt.title('Explanation for `{}`'.format(XAI.class_name(class_idx)))
           plt.axis('off')
           plt.imshow(img)
           plt.imshow(sal_exp, cmap='jet', alpha=0.5)
           plt.axis('off') 
           plt.savefig(dst_path2)
           
    
           
 elif evaluate == 'full_dataset':
     
    input_size = (224,224)
    explanations = np.empty((len(test_generator.filenames), *input_size))
    # for grad-cam model initalization
    #model = gdc.build_model()
    ## choosing the method which you want to run and evaluate on given dataset
    run  = 'my_method'
    for i in range(len(test_generator.filenames)):
        #read_path = imagelist[i]
        read_path = test_generator.filepaths[i]
        
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        img, x = XAI.load_img(read_path,(224,224))
        
        if run == 'my_method':
           ##  first getting the last convolutional layers of the model 
           conv_features = features_model.predict(x)
           last_conv_output = np.squeeze(conv_features)
           ## GENERATING THE MASK FROM THE LAST CONV LAYER OF THE GIVEN MODEL
           masks, grid, cell_size, up_size = XAI.generate_masks_conv_output((224,224), last_conv_output, 8)
           ## CHANGING THE DATA SHAPE ORDER
           new_masks = np.rollaxis(masks, 2, 0)
           size = new_masks.shape
           data = new_masks.reshape(size[0], size[1], size[2], 1)
          
           N = len(data)
           ## choose one of my method 
           my_method = 'SIDU'
          
         
           if my_method == 'SIDU':          
           ##### with  SIM diff and uniqness
                sal, weights, new_interactions, diff_interactions, pred_org = XAI.explain_SIDU(base_model, x, N, 0.5, data, (224,224))
                pred = np.argmax(pred_org)
                ## enter your floder to save the heatmaps of the given dataset
                save_path = XAI.fold_dir('./IMAGENET_MY_METHOD_HEATMAPS_SIM_DIFFF_INTERACTIONS_to_publish')
                dst_image = os.path.splitext(read_path.split('/')[-1])[0]+'_MY_METHOD_SIDU'+'.jpg'
                sal_exp = sal[pred]
                class_idx = pred
                explanations[i] = sal_exp
           
    
           
           cv2.normalize(sal_exp, sal_exp, 0, 255, cv2.NORM_MINMAX)
           dst_path = os.path.join(save_path, dst_image)
           cv2.imwrite(dst_path, sal_exp) 
           XAI._show_explanation(img, sal[class_idx], class_idx,  read_path+'imagenet', show=True, save=True, save_folder=save_path, alpha=0.3, figsize=(15, 5),
                        cmap='jet')  
           dst_plot = os.path.splitext(read_path.split('/')[-1])[0]+'_MY_METHOD_plot'+'.png'

           dst_path2 = os.path.join(save_path, dst_plot)
           plt.title('Explanation for `{}`'.format(XAI.class_name(class_idx)))
           plt.axis('off')
           plt.imshow(img)
           plt.imshow(sal_exp, cmap='jet', alpha=0.5)
           plt.axis('off') 
           plt.savefig(dst_path2) 
         
        
    
#    if run =='my_method':
#       #SAVING THE DATA IN THE .NPY FILES
#      
#       #explanations.tofile('explnations_my_method_SIDU_dataset3{:05}-{:05}.npy'.format(5000, 7000))
#
#       #exp = np.fromfile('explnations_my_method_SIDU_dataset3{:05}-{:05}.npy'.format(5000, 7000)).reshape((2000, 224, 224))
#
