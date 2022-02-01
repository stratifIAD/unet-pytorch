"""
Adaptate from https://gist.github.com/jdhao/9a86d4b9e4f79c5330d54de991461fd6
Calculate Mean and Std per channel
and standard deviation in the training set, 
suggestions: http://cs231n.github.io/neural-networks-2/#datapre
Input:images HxWxCH
Output: mean, std 
"""

import numpy as np
from os import listdir
from os.path import join, isdir
import glob 
import cv2
import timeit
from pathlib import Path

def get_dataset_stats(data_loader):
    """
    Input: data_loader #BathxCHxHxW
    Output: mean ([vector]) and std ([vector]).
    """
    train_mean = []
    train_std = []

    for i, image in enumerate(data_loader, 0):
        numpy_image = image[0].numpy()
        #print(numpy_image.shape)

        batch_mean = np.mean(numpy_image, axis=(0,2,3)) # compute batch-mean
        batch_std  = np.std(numpy_image, axis=(0,2,3)) # compute batch-std

        train_mean.append(batch_mean)
        train_std.append(batch_std)

    train_mean = torch.tensor(np.mean(train_mean, axis=0), dtype=torch.float32) # compute training mean
    train_std = torch.tensor(np.mean(train_std, axis=0), dtype=torch.float32) # compute training std
    
    return train_mean, train_std

##########
def cal_dir_stat(im_pths,CHANNEL_NUM): ##give the names 
    pixel_num = 0
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)



    for path in im_pths:
        im = np.load(path)
        im= im[:,:,:CHANNEL_NUM]
        pixel_num += (im.size/CHANNEL_NUM)
        channel_sum += np.sum(im, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))


    img_mean = channel_sum / pixel_num
    img_std = np.sqrt(channel_sum_squared / pixel_num - np.square(img_mean))
       
    return img_mean, img_std


def meanstd(im_pths_train,CHANNEL_NUM): 
 

    
    start = timeit.default_timer()
    mean_train, std_train = cal_dir_stat(im_pths_train,CHANNEL_NUM) 
    end = timeit.default_timer()
    
    print("elapsed time: {}".format(end-start))
    print("Train mean:{}\nstd:{}".format(mean_train, std_train))
  
    
    return mean_train, std_train
