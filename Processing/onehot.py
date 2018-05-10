import numpy as np
import os
import sys
import sklearn.preprocessing

def OneHot(a):
    height,width = a.shape
    a = a.flatten()
    a = a.astype(int)
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(max(a)+1))
    b = label_binarizer.transform(a)
    b = b.reshape((height,width,a.max()+1))
    return b

'''
for file in os.listdir(sys.argv[1]):
    if file.endswith(".npy"):
        directory = os.path.join(sys.argv[1], file)
        inpt = np.load(directory)
        mat = OneHot(inpt)
        out_name = directory[:-4]    
        np.save('{}'.format(out_name + 'onehot'),mat)
'''
