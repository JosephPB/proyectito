"""
If edge detection is proving difficult in detecting boxes becuase of a more continuous gradient at the edge, then could incease the contrast.
"""

import cv2
import numpy as np

im = cv2.imread('test.jpg')
cp = im

height, width, depth = im.shape

for i in range(height):
    for j in range(width):
        r,g,b = cp[i,j]
        
        #greyscale picture so RGB values are all equal
        if r < 30:
            cp[i,j] = [1,1,1]

cv2.imwrite('testaltered.jpg', cp)

im2 = cv2.imread('wrist.jpg')

height, width, depth = im2.shape
cp = im2

for i in range(height):
    for j in range(width):
        r,g,b = cp[i,j]
        
        #greyscale picture so RGB values are all equal
        
        #altering darker pixels
        if r < 30:
            cp[i,j] = [1,1,1]

        #altering lighter pixels
        if r >= 30:
            cp[i,j] = [255,255,255]

cv2.imwrite('wristaltered.jpg', im2)
