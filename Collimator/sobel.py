import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('/media/sf_TestImageDataBase/rectangle/jpeg_converted/Wrist.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(img, cmap = 'gray')
plt.colorbar()
plt.show()

# Gradient in x
sobelx64f = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=1)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)
plt.imshow(sobel_8u)
plt.show()

minLineLength = 1000
maxLineGap = 800

lines = cv2.HoughLinesP(sobel_8u, 1, np.pi/180,1500,minLineLength,maxLineGap)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),3)

plt.imshow(img)
plt.show()

