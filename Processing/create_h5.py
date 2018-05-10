from __future__ import division
import os, sys
import numpy as np
import cv2
import glob
from random import shuffle
from IPython.display import clear_output
import h5py
from sklearn.model_selection import train_test_split
import label as label_generate
import onehot


def write_h5(hdf5_name,images,labels):
    train_size = 0.7
    images_train, images_test, labels_train, labels_test = train_test_split(\
            images, labels, test_size = 1.-train_size, random_state=42)

    images_test, images_val, labels_test, labels_val = train_test_split(\
            images_test, labels_test, test_size = 0.5, random_state=42)
 
    hdf5_file = h5py.File(hdf5_name, mode='w')
    # Attributes
    hdf5_file.attrs['image_size'] = images.shape[2] 
    hdf5_file.attrs['max_value'] = 1.
    hdf5_file.attrs['min_value'] = 0.

    # Datasets
    hdf5_file.create_dataset("train_img", images_train.shape, np.float64)
    hdf5_file.create_dataset("train_label", labels_train.shape, np.float64)

    hdf5_file.create_dataset("test_img", images_test.shape, np.float64)
    hdf5_file.create_dataset("test_label", labels_test.shape, np.float64)
 
    hdf5_file.create_dataset("val_img", images_val.shape, np.float64)
    hdf5_file.create_dataset("val_label", labels_val.shape, np.float64)
 
    categories = ['train','test','val']
    images_split = [images_train, images_test, images_val]
    labels_split =  [labels_train, labels_test, labels_val]
    for j  in range(len(images_split)):
        for i in range(images_split[j].shape[0]):
            clear_output(wait=True)
            # Normalization -> perform after augmentation
            img = images_split[j][i,...]/np.max(images_split[j][i,...])
            hdf5_file[categories[j] + '_img'][i, ...] = img

            # same for labels
            #labels_simple = label_generate.GenerateOutput(labels_split[j][i,...])
            #labels_onehot = onehot.OneHot(labels_simple)
            #hdf5_file[categories[j] + "_label"][i, ...] = labels_onehot    
            hdf5_file[categories[j] + "_label"][i, ...] = labels_split[j][i,...] 
            print('Saving image %i/%i in %s path' %(i+1,images_split[j].shape[0], categories[j]))

    hdf5_file.close() 
