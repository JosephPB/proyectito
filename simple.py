import cv2
import matplotlib.pyplot as plt
import numpy as np


im = cv2.imread('wrist.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

contrast = cv2.imread('wristaltered.jpg')

gray = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)


mask,contours,hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

print(mask)

cnt = contours[-1]
x,y,w,h =cv2.boundingRect(cnt)

crop = im[y:y+h,x:x+w]
plt.imshow(crop)
plt.show()

cv2.imwrite('crop.jpg',crop)


