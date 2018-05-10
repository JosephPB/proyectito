"""
BGR value of white = [255,255,255], of cyan = [255,255,0], of red = [0,0,254]
input: command line input

e.g. for f in *.jpg; do  echo "Converting $f"; convert "$f"  "$(basename "$f" .jpg).npy"; done

output:
 - for bone (red): 2
 - for soft tissue (cyan): 1
 - for open beam (white): 0

To reduce memory issues, creates a sub-array which appends to the main array once it reaches 100 entries.

"""


import cv2 as cv
import numpy as np
import sys
import os

def GenerateOutput(image):

    #image = cv.imread(inpt)

    otpt = np.array([]) #sub-array
    out = np.array([]) #main output array
    
    height,width,depth = image.shape

    counter = 0 #set counter for sub-array
    for i in range(height):
        for j in range(width):
            b,g,r = image[i,j]

            #identify red region
            if r > 220 and b < 30 and g < 30:
                otpt = np.append(otpt,2)

            #identify cyan region
            elif b > 220 and g > 220 and r < 30:
                otpt = np.append(otpt,1)
                
            #identify white region    
            elif r > 220  and g > 220 and b > 220:
                otpt = np.append(otpt,0)

            #if criterion aren't met, flag error    
            else:
                print ('pixel not classified!')
                sys.exit("unclassified pixed has BGR values of {},{},{}".format(b,g,r))

            #append sub-array and reset once it gets to 100 entries
            #avoid problem of total pixel no not being % 100, only reset up to final line
            counter += 1
            if counter == 100 and i != height-1:
                out = np.append(out,otpt)
                out.flatten()
                otpt = np.array([])
                counter = 0                
                
        #print ('appended line {} of {}'.format(i,height))

    #if total pixel no is a multiple of 100
    if len(otpt) != 0:    
        out = np.append(out,otpt)
        out.flatten()

    #pixel classification count    
    red = np.count_nonzero(out == 2)
    cyan = np.count_nonzero(out == 1)
    white = height*width - (red+cyan) 
    #print ('{} red pixels classified'.format(red))
    #print ('{} cyan pixels classified'.format(cyan))
    #print ('{} white pixels classified'.format(white))

    out = np.reshape(out,(height,width))
    
    if len(out[0]) == width and len(out) == height:
        print ('height and width of labels match image dimensions')

    return out 

"""
if __name__ == '__main__':

    '''
    inpt = sys.argv[1]
            
    GenerateOutput(inpt)
    '''

    for file in os.listdir(sys.argv[1]):
        if file.endswith(".jpg"):
            directory = os.path.join(sys.argv[1], file)
            print (directory)
            GenerateOutput(directory)
"""
