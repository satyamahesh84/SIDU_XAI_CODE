# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 21:55:03 2018

@author: user
"""

import cv2
import numpy as np
from tqdm import tqdm
import csv
import pandas as pd
from matplotlib import pyplot as plt
import os
from pathlib import Path
from PIL import Image

def GaussianMask(sizex,sizey, sigma=33, center=None,fix=1):
    """
    sizex  : mask width
    sizey  : mask height
    sigma  : gaussian Sd
    center : gaussian mean
    fix    : gaussian max
    return gaussian mask
    """
    x = np.arange(0, sizex, 1, float)
    y = np.arange(0, sizey, 1, float)
    x, y = np.meshgrid(x,y)
    
    if center is None:
        x0 = sizex // 2
        y0 = sizey // 2
    else:
        if np.isnan(center[0])==False and np.isnan(center[1])==False:            
            x0 = center[0]
            y0 = center[1]        
        else:
            return np.zeros((sizey,sizex))

    return fix*np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)

def Fixpos2Densemap(fix_arr, width, height, imgfile, plot_on_image = True, alpha=0.5, threshold=10):
    """
    fix_arr   : fixation array number of subjects x 3(x,y,fixation)
    width     : output image width
    height    : output image height
    imgfile   : image file (optional)
    alpha     : marge rate imgfile and heatmap (optional)
    threshold : heatmap threshold(0~255)
    return heatmap 
    """
    H = height
    W = width
    heatmap = np.zeros((H,W), np.float32)
    for n_subject in tqdm(range(fix_arr.shape[0])):
#        heatmap += GaussianMask(W, H, 33, (fix_arr[n_subject,0],fix_arr[n_subject,1]),
#                                fix_arr[n_subject,2])
        heatmap += GaussianMask(W, H, 70, (fix_arr[n_subject,0],fix_arr[n_subject,1]))

    # Normalization
    heatmap = heatmap/np.amax(heatmap)
    heatmap = heatmap*255
    heatmap = heatmap.astype("uint8")
    
    if  plot_on_image == True:                                                #imgfile.any():
        # Resize heatmap to imgfile shape 
        h, w, _ = imgfile.shape
        heatmap = cv2.resize(heatmap, (w, h))
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Create mask
        mask = np.where(heatmap<=threshold, 1, 0)
        mask = np.reshape(mask, (h, w, 1))
        mask = np.repeat(mask, 3, axis=2)

        # Marge images
        marge = imgfile*mask + heatmap_color*(1-mask)
        marge = marge.astype("uint8")
        marge = cv2.addWeighted(imgfile, 1-alpha, marge,alpha,0)
        return marge

    else:
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap

def KLdiv(saliencyMap, fixationMap):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map

    # convert to float
    map1 = saliencyMap.astype(float)
    map2 = fixationMap.astype(float)

    # make sure maps have the same shape
    from scipy.misc import imresize
    map1 = imresize(map1, np.shape(map2))
    #map1 = np.array(Image.fromarray(im).resize((h, int(w * aspect_ratio))))

    # make sure map1 and map2 sum to 1
    if map1.any():
        map1 = map1 / map1.sum()
    if map2.any():
        map2 = map2 / map2.sum()

    # compute KL-divergence
    eps = 10 ** -12
    score = map2 * np.log(eps + map2 / (map1 + eps))

    return score.sum()


def AUC_Judd(saliencyMap, fixationMap, jitter=True, toPlot=False):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map (binary matrix)
    # jitter=True will add tiny non-zero random constant to all map locations to ensure
    # 		ROC can be calculated robustly (to avoid uniform region)
    # if toPlot=True, displays ROC curve

    # If there are no fixations to predict, return NaN
    if not fixationMap.any():
        print('Error: no fixationMap')
        score = float('nan')
        return score

    # make the saliencyMap the size of the image of fixationMap

    if not np.shape(saliencyMap) == np.shape(fixationMap):
        from scipy.misc import imresize
        saliencyMap = imresize(saliencyMap, np.shape(fixationMap))

    # jitter saliency maps that come from saliency models that have a lot of zero values.
    # If the saliency map is made with a Gaussian then it does not need to be jittered as
    # the values are varied and there is not a large patch of the same value. In fact
    # jittering breaks the ordering in the small values!
    if jitter:
        # jitter the saliency map slightly to distrupt ties of the same numbers
        saliencyMap = saliencyMap + np.random.random(np.shape(saliencyMap)) / 10 ** 7

    # normalize saliency map
    saliencyMap = (saliencyMap - saliencyMap.min()) \
                  / (saliencyMap.max() - saliencyMap.min())

    if np.isnan(saliencyMap).all():
        print('NaN saliencyMap')
        score = float('nan')
        return score

    S = saliencyMap.flatten()
    F = fixationMap.flatten()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)
    Npixels = len(S)

    allthreshes = sorted(Sth, reverse=True)  # sort sal map values, to sweep through values
    tp = np.zeros((Nfixations + 2))
    fp = np.zeros((Nfixations + 2))
    tp[0], tp[-1] = 0, 1
    fp[0], fp[-1] = 0, 1

    for i in range(Nfixations):
        thresh = allthreshes[i]
        aboveth = (S >= thresh).sum()  # total number of sal map values above threshold
        tp[i + 1] = float(i + 1) / Nfixations  # ratio sal map values at fixation locations
        # above threshold
        fp[i + 1] = float(aboveth - i) / (Npixels - Nfixations)  # ratio other sal map values
        # above threshold

    score = np.trapz(tp, x=fp)
    allthreshes = np.insert(allthreshes, 0, 0)
    allthreshes = np.append(allthreshes, 1)

    if toPlot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.matshow(saliencyMap, cmap='gray')
        ax.set_title('SaliencyMap with fixations to be predicted')
        [y, x] = np.nonzero(fixationMap)
        s = np.shape(saliencyMap)
        plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
        plt.plot(x, y, 'ro')

        ax = fig.add_subplot(1, 2, 2)
        plt.plot(fp, tp, '.b-')
        ax.set_title('Area under ROC curve: ' + str(score))
        plt.axis((0, 1, 0, 1))
        plt.show()

    return score

def AUC_Judd_tp_fp(saliencyMap, fixationMap, jitter=True, toPlot=False):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map (binary matrix)
    # jitter=True will add tiny non-zero random constant to all map locations to ensure
    # 		ROC can be calculated robustly (to avoid uniform region)
    # if toPlot=True, displays ROC curve

    # If there are no fixations to predict, return NaN
    if not fixationMap.any():
        print('Error: no fixationMap')
        score = float('nan')
        return score

    # make the saliencyMap the size of the image of fixationMap

    if not np.shape(saliencyMap) == np.shape(fixationMap):
        from scipy.misc import imresize
        saliencyMap = imresize(saliencyMap, np.shape(fixationMap))

    # jitter saliency maps that come from saliency models that have a lot of zero values.
    # If the saliency map is made with a Gaussian then it does not need to be jittered as
    # the values are varied and there is not a large patch of the same value. In fact
    # jittering breaks the ordering in the small values!
    if jitter:
        # jitter the saliency map slightly to distrupt ties of the same numbers
        saliencyMap = saliencyMap + np.random.random(np.shape(saliencyMap)) / 10 ** 7

    # normalize saliency map
    saliencyMap = (saliencyMap - saliencyMap.min()) \
                  / (saliencyMap.max() - saliencyMap.min())

    if np.isnan(saliencyMap).all():
        print('NaN saliencyMap')
        score = float('nan')
        return score

    S = saliencyMap.flatten()
    F = fixationMap.flatten()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)
    Npixels = len(S)

    allthreshes = sorted(Sth, reverse=True)  # sort sal map values, to sweep through values
    tp = np.zeros((Nfixations + 2))
    fp = np.zeros((Nfixations + 2))
    tp[0], tp[-1] = 0, 1
    fp[0], fp[-1] = 0, 1

    for i in range(Nfixations):
        thresh = allthreshes[i]
        aboveth = (S >= thresh).sum()  # total number of sal map values above threshold
        tp[i + 1] = float(i + 1) / Nfixations  # ratio sal map values at fixation locations
        # above threshold
        fp[i + 1] = float(aboveth - i) / (Npixels - Nfixations)  # ratio other sal map values
        # above threshold

    score = np.trapz(tp, x=fp)
    allthreshes = np.insert(allthreshes, 0, 0)
    allthreshes = np.append(allthreshes, 1)

    if toPlot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.matshow(saliencyMap, cmap='gray')
        ax.set_title('SaliencyMap with fixations to be predicted')
        [y, x] = np.nonzero(fixationMap)
        s = np.shape(saliencyMap)
        plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
        plt.plot(x, y, 'ro')

        ax = fig.add_subplot(1, 2, 2)
        plt.plot(fp, tp, '.b-')
        ax.set_title('Area under ROC curve: ' + str(score))
        plt.axis((0, 1, 0, 1))
        plt.show()

    return score, tp, fp

def discretize_gt(gt):
	import warnings
	warnings.warn('can improve the way GT is discretized')
	return gt/255

def auc_borji(s_map,gt,splits=100,stepsize=0.1):
	gt = discretize_gt(gt)
	num_fixations = np.sum(gt)

	num_pixels = s_map.shape[0]*s_map.shape[1]
	random_numbers = []
	for i in range(0,splits):
		temp_list = []
		for k in range(0,int(num_fixations)):
			temp_list.append(np.random.randint(num_pixels))
		random_numbers.append(temp_list)

	aucs = []
	# for each split, calculate auc
	for i in random_numbers:
		r_sal_map = []
		for k in i:
			r_sal_map.append(s_map[k % s_map.shape[0]-1, k/s_map.shape[0]])
		# in these values, we need to find thresholds and calculate auc
		thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

		r_sal_map = np.array(r_sal_map)

		# once threshs are got
		thresholds = sorted(set(thresholds))
		area = []
		area.append((0.0,0.0))
		for thresh in thresholds:
			# in the salience map, keep only those pixels with values above threshold
			temp = np.zeros(s_map.shape)
			temp[s_map>=thresh] = 1.0
			num_overlap = np.where(np.add(temp,gt)==2)[0].shape[0]
			tp = num_overlap/(num_fixations*1.0)
			
			#fp = (np.sum(temp) - num_overlap)/((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)
			# number of values in r_sal_map, above the threshold, divided by num of random locations = num of fixations
			fp = len(np.where(r_sal_map>thresh)[0])/(num_fixations*1.0)

			area.append((round(tp,4),round(fp,4)))
		
		area.append((1.0,1.0))
		area.sort(key = lambda x:x[0])
		tp_list =  [x[0] for x in area]
		fp_list =  [x[1] for x in area]

		aucs.append(np.trapz(np.array(tp_list),np.array(fp_list)))
	
	return np.mean(aucs), tp_list, fp_list

def fold_dir(folder):
    if not os.path.exists(folder):
           os.makedirs(folder)
    return folder
if __name__ == '__main__':
    # Load image file
#    img = cv2.imread('sample.png')
#    
#    # Generate toy fixation data
#    # when you use, replace here with your data
#    num_subjects = 40
#    H, W, _ = img.shape
#    
#    fix_arr = np.random.randn(num_subjects,3)
#    fix_arr -= fix_arr.min()
#    fix_arr /= fix_arr.max()
#    fix_arr[:,0] *= W
#    fix_arr[:,1] *= H
#    
#    # Create heatmap
#    heatmap = Fixpos2Densemap(fix_arr, W, H, img, 0.7, 5)
#    cv2.imwrite("output.png",heatmap)
    
 input_path = '/home/administrator/GazePointHeatMap/Example_Output/eye_data.csv'
# display_width = 1440
# display_height = 900
 display_width = 1920
 display_height = 1114
 alpha = 0.6
 output_name = 'output-new'
# background_image = '/home/administrator/GazePointHeatMap/Example_Output/02.jpg'
 background_image= 'test_o2.jpg'
 cnn_saliency_image = '02_CAM.jpg'
 ngaussian = 200
 sd = 33  
 dim = (1920, 1114)
 img = cv2.imread(background_image)
 img = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
 
 img_sal = cv2.imread(cnn_saliency_image)
 img_sal = cv2.resize(img_sal, dim, interpolation = cv2.INTER_CUBIC)
 gray_scale_sal = cv2.cvtColor(img_sal,cv2.COLOR_BGR2GRAY)
 cv2.normalize(gray_scale_sal,gray_scale_sal, 0, 255, cv2.NORM_MINMAX)
 cv2.imwrite("output_new_inter_cubic_new_Test_1_cnn_sak.png",gray_scale_sal)

 #    # Generate toy fixation data
    # when you use, replace here with your data
 num_subjects = 40
 H, W, _ = img.shape  
 raw = pd.read_csv(input_path)  
 #data = raw.iloc[:, 0:2]  
 fix_arr_new = np.zeros((237, 2))
 fix_arr_new[:,0] = raw.iloc[:,0]
 fix_arr_new[:,1] = raw.iloc[:,1]
 
# #    # Create heatmap
 heatmap = Fixpos2Densemap(fix_arr_new, display_width, display_height, img, False, 0.7, 10)
# cv2.normalize(heatmap,heatmap, 0, 255, cv2.NORM_MINMAX)
 #dst_path = os.path.join(save_path, dst_image)
# cv2.imwrite(dst_path, final_output) 
 cv2.imwrite("output_new_inter_cubic_new_Test_1.png",heatmap)
 plt.imshow(heatmap, cmap='jet')                           
 plt.axis('off')
 plt.show()
#cv2.imshow('heatmap',heatmap)
#cv2.waitKey()
## 
   
## KL DIVERGECE EVALAUTION
 
 gray_scale_sal = gray_scale_sal.astype('float32')
 cv2.normalize(gray_scale_sal,gray_scale_sal, 0, 1, cv2.NORM_MINMAX)
 fix_sal = cv2.cvtColor(heatmap,cv2.COLOR_BGR2GRAY)
 fix_sal_eye = fix_sal.astype('float32')
 cv2.normalize(fix_sal_eye,fix_sal_eye, 0, 1, cv2.NORM_MINMAX)
 KLdiv(gray_scale_sal, fix_sal_eye)