import cv2
import matplotlib.pyplot as plt
import numpy as np


im = cv2.imread('wrist.jpg')
#im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

#contrast = cv2.imread('wristaltered.jpg')

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

limit = 500

closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
            np.ones((limit,limit), np.uint8))

plt.imshow(closed)
plt.show()

edges = cv2.Canny(closed,0,50,apertureSize = 3)

plt.imshow(edges)
plt.show()

w_threshold =limit  

h_threshold =limit 

mask,contours,hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

valid_contours = []
valid_w = []
valid_h = []
out = np.zeros_like(im)
for cnt in contours:
    area = cv2.contourArea(cnt)
    #x,y,w,h = cv2.boundingRect(cnt)
    if(area >= limit*limit):
        cv2.drawContours(out, cnt, -1, 255,3)
        plt.imshow(out)
        plt.show()
        valid_contours.append(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        crop = im[y:y+h,x:x+w]
        plt.imshow(crop)
        plt.show()

'''
cnt = contours[4]

x,y,w,h =cv2.boundingRect(cnt)

crop = im[y:y+h,x:x+w]

plt.imshow(crop)
plt.show()

cv2.imwrite('crop.jpg',crop)

'''
