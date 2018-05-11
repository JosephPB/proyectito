"""
Writes hdf5 file from data and labels folders into an hdf5 file.

How to use:

    python generate_data.py -Naug 100 -imsize 200 -h5name test

Output:
    hdf5 file splitted into train, test and validation

"""

from __future__ import division
from sklearn.model_selection import train_test_split
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import rasterio
import create_h5
import cv2
import glob
from random import shuffle
import h5py
import argparse
import label as label_generate
import onehot
import scipy.misc
from shutil import copyfile


main_path = 'X:\company-public\Projects - Internal IBEX projects\ImageSegmentationCTD\TestImageDataBase\\training\\'
# Define command line arguments
parser = argparse.ArgumentParser(description='Save training images and labels in a hdf5 file.')

parser.add_argument('-Naug', dest='EXAMPLES_PER_CATEGORY', type=int, default=0,
                   help='Augment every bodypart to Naug images.')

parser.add_argument('-imsize', dest='image_size', type=int, default=200, 
                   help='Size of final images.')

parser.add_argument('-h5name', dest='hdf5_name', type=str, 
                   help='Name of hdf5 output file.')


args = parser.parse_args()


if(args.EXAMPLES_PER_CATEGORY == 0):
    hdf5_name = args.hdf5_name + '.hdf5'
    #hdf5_name = '/media/sf_training/hdf5/'+args.hdf5_name + '.hdf5'
else:
    hdf5_name = args.hdf5_name +'_'+ str(args.EXAMPLES_PER_CATEGORY)+ '.hdf5'
    #hdf5_name = '/media/sf_training/hdf5/'+args.hdf5_name +'_'+ str(args.EXAMPLES_PER_CATEGORY)+'.hdf5'

#data_path = '/media/sf_training/data/*.tif'
#labels_path = '/media/sf_training/labels/OneHot/*.npy'

data_path = main_path + 'data\*.tif'
labels_path = main_path + 'labels\Images\*hot.npy'

images = glob.glob(data_path)
labels =  glob.glob(labels_path)

print('Checking if number of labeled files matches number of data image files....')
# Check that number of labels corresponds to number of images
assert len(labels) == len(images)

print('Check that labels match data ....')
# Check that they have the same names
for (i, img) in enumerate(images):
    label_filename = labels[i].split('\\')[-1].split('.')[0] 
    img_filename  = img.split('\\')[-1].split('.')[0] + 'onehot'
    assert label_filename == img_filename

print('Names of labels and data match perfectly, good job :)')

#shuffle data
c = list(zip(images,labels))
shuffle(c)
images, labels = zip(*c)


# Read and save all images + labels + bodypart

images_read = np.zeros((len(images),args.image_size,args.image_size,1),dtype=np.float32)
labels_read = np.zeros((len(labels), args.image_size, args.image_size,3))
bodyparts = np.empty((len(images)),'S10')
for i in range(len(images)):
    filename = images[i]
    img = rasterio.open(filename)
    img = img.read(1)
    images_read[i,...,0] = cv2.resize(img, (args.image_size, args.image_size), interpolation=cv2.INTER_CUBIC)
    # Normalize images from 0 to 1
    #images_read[i,...,0] /= np.max(images_read[i,...,0])
    label_filename = labels[i]
    #label = cv2.imread(label_filename)
        #labels_read[i,...] = cv2.resize(label, (args.image_size, args.image_size), interpolation=cv2.INTER_NEAREST)
    #labels_read[i,...] = scipy.misc.imresize(label, (args.image_size, args.image_size,3), interp= 'nearest', mode=None)
   
    #label_onehot = label_generate.GenerateOutput(label)

    #label_onehot = onehot.OneHot(label_onehot)

    label_onehot = np.load(label_filename)

    labels_read[i,...] = scipy.misc.imresize(label_onehot, (args.image_size,args.image_size,3), interp='nearest', mode=None)


    bodypart = filename.split('\\')[-1].split('_')[0].lower()
    if((bodypart == 'left') or (bodypart == 'right') or (bodypart == 'asg')):
        bodypart = filename.split('\\')[-1].split('_')[1]
        if(bodypart == 'fractured'):
            bodypart = filename.split('\\')[-1].split('_')[2]
        if(bodypart == 'lower'):
            bodypart = filename.split('\\')[-1].split('_')[2]
        
    # Remove numbers
    bodypart = ''.join(i for i in bodypart if not i.isdigit())
    if(bodypart == 'nof'):
        bodypart = 'neckoffemur'
    bodypart = bodypart.split('.')[0]
    if(bodypart == 'anke'):
        bodypart = 'ankle'
    bodypart = bodypart.encode("ascii", "ignore")
    bodyparts[i] = bodypart



if(args.EXAMPLES_PER_CATEGORY == 0):
    create_h5.write_h5(hdf5_name, images_read,labels_read)
else:
    unique, counts = np.unique(bodyparts, return_counts=True)
    unique_per_category = dict(zip(unique, counts))
    print(unique_per_category)
    augmentations_per_category = dict(unique_per_category)
    for key in unique_per_category:
        augmentations_per_category[key] = int(args.EXAMPLES_PER_CATEGORY/unique_per_category[key])
    print(augmentations_per_category)
    

    print('type is')
    print(type(images_read[0,0,0,0]))
    #Augmentation templates
    translate_max = 0.1
    rotate_max = 45
    shear_max = 10

    affine_trasform = iaa.Affine( translate_percent={"x": (-translate_max, translate_max),
                                                     "y": (-translate_max, translate_max)}, # translate by +-
                                  rotate=(-rotate_max, rotate_max), # rotate by -rotate_max to +rotate_max degrees
                                  shear=(-shear_max, shear_max), # shear by -shear_max to +shear_max degrees
                                  order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                                  cval=0, # if mode is constant, use a cval between 0 and 255
                                  mode="edge",
                                  name="Affine",
                                 )

    spatial_aug = iaa.Sequential([iaa.Crop(percent=(0, 0.2)),iaa.Fliplr(0.5), iaa.Flipud(0.5), affine_trasform])

    other_aug = iaa.SomeOf((0, None),
            [
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 1.0
                    iaa.AverageBlur(k=(2,5)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # few
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), #  few
                    iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5), #  few
                    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25), # very few
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2)

                ]),

            ])

            

    augmentator = [spatial_aug,other_aug]
    total_images=sum(augmentations_per_category[k]*unique_per_category[k] for k in augmentations_per_category)
    images_aug = np.zeros((total_images,images_read.shape[1],images_read.shape[2],images_read.shape[3]))
    labels_aug = np.zeros((total_images,labels_read.shape[1],labels_read.shape[2],labels_read.shape[3]))
    bodypart = np.empty((total_images),dtype = 'S10')
    # Loop  over the different kind of bodyparts
    counter = 0
    counter_block = 0
    for i, (k, v) in enumerate(augmentations_per_category.items()):
        # Indices of images with a given bodypart
        indices = np.array(np.where(bodyparts == k )[0])
        counter_block += len(indices)
        # Number of augmentation per image
        N = int(v)
        
        for j in indices:
            for l in range(N):
                clear_output(wait=True)
                # Freeze randomization to apply same to labels
                spatial_det = augmentator[0].to_deterministic() 
                other_det = augmentator[1]
                images_aug[counter,...] = spatial_det.augment_image(images_read[j])
                images_aug[counter,...] = other_det.augment_image(images_aug[counter,...])
                labels_aug[counter,...] = spatial_det.augment_image(labels_read[j])
                labels_aug[counter,...] = np.rint(labels_aug[counter,...]/255)

                bodypart[counter] = k
                print('(Category %s) processing image %i/%i, augmented image %i/%i'%(k,counter_block ,
                                                                                         bodyparts.shape[0],
                                                                                         l+1, N))
                counter +=1
                
    print('Finished playing with cadavers ! ')
    create_h5.write_h5(hdf5_name, images_aug,labels_aug)
copyfile(hdf5_name, main_path + 'hdf5\\' + hdf5_name)
os.remove(hdf5_name)
