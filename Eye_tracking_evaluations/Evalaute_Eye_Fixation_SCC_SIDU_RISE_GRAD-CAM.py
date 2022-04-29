#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:08:39 2020

@author: administrator
"""

import Fixpos2Densemap as FD
import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import glob
import SIDU_XAI as XAI
from pathlib import Path
from os.path import join
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os.path import join, exists
from PIL import ImageFile
from scipy.stats import spearmanr

if __name__ == '__main__':

 run = 'all_images'

 base_dir = 'Eye_Tracking_Evalautions'
 

 eye_heatmaps_dir = os.path.join(base_dir, 'Groundtruth_eye')
 
 ## this is alternate way of reading the data from the folder
 eyetracker = sorted(glob.glob('Eye_Tracking_Evalautions/Groundtruth_eye/EyeTrackerMasks/*.jpg'))
 
 ## FOR resent
 # gradcam = sorted(glob.glob('Eye_Tracking_Evalautions/validate_gradcam/GRAD-CAM-HEATMAPS/*.jpg'))
 # rise = sorted(glob.glob('Eye_Tracking_Evalautions/validate_rise/RISE_HEATMAPS/*.jpg'))
 # sidu = sorted(glob.glob('Eye_Tracking_Evalautions/validate_sidu/SIDU_HEATMAPS/*.jpg'))

 ## for mobilenet
 gradcam = sorted(glob.glob('Eye_Tracking_Evalautions/validate_gradcam_mobilenet/GRAD_masks/*.jpg'))
 rise = sorted(glob.glob('Eye_Tracking_Evalautions/validate_rise_mobilenet/RISE_masks/*.jpg'))
 sidu = sorted(glob.glob('Eye_Tracking_Evalautions/validate_sidu_mobilenet/SIDU_masks/*.jpg'))


### this is the another way of reading the file from the folder using imagedatagenerator (tensorflow.keras)
 eye_test_datagen = ImageDataGenerator(rescale=1./255)
 eye_test_generator = eye_test_datagen.flow_from_directory(
     directory= eye_heatmaps_dir,
     target_size=(224, 224),
     color_mode="rgb",
     batch_size=1,
     class_mode=None,
     shuffle=False)
 
 # #method_heatmaps_dir = os.path.join(base_dir, 'validate_gradcam') # grada-cam heatmaps
 # method_heatmaps_dir = os.path.join(base_dir, 'validate_sidu')  # SIDU heatmaps
 # ##method_heatmaps_dir = os.path.join(base_dir, 'validate_rise')  # RISE heatmaps
 
 # ### this directorys contains mobilenet  heats maps for the xai methods (master thesis work)
 # #method_heatmaps_dir = os.path.join(base_dir, 'validate_gradcam_mobilenet') # grada-cam heatmaps
 # #method_heatmaps_dir = os.path.join(base_dir, 'validate_sidu_mobilenet')  # SIDU heatmaps
 # method_heatmaps_dir = os.path.join(base_dir, 'validate_rise_mobilenet')  # RISE heatmaps
 
 ### this directorys contains mobilenet  heats maps for the xai methods (master thesis work)
 #method_heatmaps_dir = os.path.join(base_dir, 'validate_gradcam_vgg16') # grada-cam heatmaps
 #method_heatmaps_dir = os.path.join(base_dir, 'validate_sidu_vgg16')  # SIDU heatmaps
 method_heatmaps_dir = os.path.join(base_dir, 'validate_rise_vgg16')  # RISE heatmaps
 
 
 
 method_test_datagen = ImageDataGenerator(rescale=1./255)
 method_test_generator = method_test_datagen.flow_from_directory(
     directory= method_heatmaps_dir,
     target_size=(224, 224),
     color_mode="rgb",
     batch_size=1,
     class_mode=None,
     shuffle=False)
 
 ## KL DIVERGECE EVALAUTION
 
 if run == 'single_image':
    eval_method = 'SCC'
    #read_path = eye_test_generator.filepaths[0]
    read_path = eyetracker[98]
    #read_img_path = method_test_generator.filepaths[0]
    read_img_path = gradcam[98]
    print('cnn_sal_image:',read_img_path)
    print('eye_fix_sal:',read_path)
    cnn_sal = cv2.imread(read_img_path)
    eye_sal = cv2.imread(read_path)
    gray_cnn_sal = cv2.cvtColor(cnn_sal,cv2.COLOR_BGR2GRAY)
    gray_eye_sal = cv2.cvtColor(eye_sal,cv2.COLOR_BGR2GRAY)
    #gray_cnn_sal = gray_cnn_sal.astype('float')
    cv2.normalize(gray_cnn_sal,gray_cnn_sal, 0, 1, cv2.NORM_MINMAX)
    #gray_eye_sal = gray_eye_sal.astype('float')
    cv2.normalize(gray_eye_sal,gray_eye_sal, 0, 1, cv2.NORM_MINMAX)
    from scipy import ndimage
    img_fill_holes=ndimage.binary_fill_holes(gray_eye_sal)
    plt.imshow(img_fill_holes)
    if eval_method == 'KL_DIV':
       klval = FD.KLdiv(gray_cnn_sal, img_fill_holes)
       print(klval)
    elif eval_method == 'SCC':
       norm_eye_map_flat = img_fill_holes.flatten()  ## eye_fixations
       norm_xai_map_flat = gray_cnn_sal.flatten()    ## xai method 
       coef, p = spearmanr(norm_eye_map_flat, norm_xai_map_flat) 
       print(coef)
       
 elif run == 'all_images':

    SCC_ALL= []
      
 
    
    run  = 'RISE'
    for i in range(len(eye_test_generator.filenames)):
        eye_img_path = eye_test_generator.filepaths[i]
        method_img_path = method_test_generator.filepaths[i]
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        print('cnn_sal_image:',method_img_path)
        print('eye_fix_sal:',eye_img_path)
        eye_sal = cv2.imread(eye_img_path)
        cnn_sal = cv2.imread(method_img_path)
        gray_cnn_sal = cv2.cvtColor(cnn_sal,cv2.COLOR_BGR2GRAY)
        gray_eye_sal = cv2.cvtColor(eye_sal,cv2.COLOR_BGR2GRAY)
        #gray_cnn_sal = gray_cnn_sal.astype('float')
        cv2.normalize(gray_cnn_sal,gray_cnn_sal, 0, 1, cv2.NORM_MINMAX)
        #gray_eye_sal = gray_eye_sal.astype('float')
        cv2.normalize(gray_eye_sal,gray_eye_sal, 0, 1, cv2.NORM_MINMAX)
        from scipy import ndimage
        img_fill_holes=ndimage.binary_fill_holes(gray_eye_sal).astype(int)  ## this fill the hole of eye_fixations
        
        ####### to save the eye_fixations with filling the holes ground_truth  binary maps
        # save_path = XAI.fold_dir('./Eye_fixations_BInary_hole_filled_HEATMAPS_Order_Changed')
        # dst_image = os.path.splitext(eye_img_path.split('/')[-1])[0]+'_Eye_fixations_binary'+'.jpg'
        # dst_path = os.path.join(save_path, dst_image)
        # plt.axis('off')
        # plt.imshow(img_fill_holes)
        # plt.axis('off') 
        # plt.savefig(dst_path, bbox_inches='tight', pad_inches=0, dpi= 74.5)
        
        norm_eye_map_flat = img_fill_holes.flatten()  ## eye_fixation_after_filling holes
        norm_xai_cnn_map_flat = gray_cnn_sal.flatten() ## xai method
        coef, p = spearmanr(norm_eye_map_flat, norm_xai_cnn_map_flat) 
        print(coef)
        SCC_ALL.append(coef)
        
        
    all_img_scc = np.array(SCC_ALL)
       
       
    if run == 'RISE':
        print('RISE_mean_SCC:',all_img_scc.mean())
        #all_img_scc.tofile('RISE_SCC_evalaution_Resenet-50_MASTER_THESIS_EYE_FIXATIONS.npy')
        all_img_scc.tofile('RISE_SCC_evalaution_vgg16_MASTER_THESIS_EYE_FIXATIONS.npy')

    elif run == 'SIDU' :
        print('SIDU_mean_SCC:',all_img_scc.mean())
        #all_img_scc.tofile('SIDU_SCC_evalaution_Resnet-50_MASTER_THESIS_EYE_FIXATIONS.npy')
        all_img_scc.tofile('SIDU_SCC_evalaution_vgg16_MASTER_THESIS_EYE_FIXATIONS.npy')

    elif run == 'GRAD-CAM' :
        print('GRAD-CAM_mean_SCC:',all_img_scc.mean())
        #all_img_scc.tofile('GRAD-CAM_SCC_evalaution_Resnet-50_MASTER_THESIS_EYE_FIXATIONS.npy')
        all_img_scc.tofile('GRAD-CAM_SCC_evalaution_vgg16_MASTER_THESIS_EYE_FIXATIONS.npy')