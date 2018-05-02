import cv2
import numpy as np
import argparse
import glob

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

img = cv2.imread('Wrist.jpg')
backtorgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
cv2.imwrite('color.jpg',backtorgb)
# Extract edges
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.ones((500, 500), np.uint8)
closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('closing.jpg',closing)
edges = cv2.Canny(closing,225,250,apertureSize = 3)
#edges = auto_canny(closing)
cv2.imwrite('edges.jpg',edges)

# Try to automate threshold to select only three lines,
# need to implement reject when two lines are too close to each other
'''
thresholds = np.arange(100,250,10)
lines = []
c = 0
while(len(lines) != 3):
        lines = cv2.HoughLines(edges,1,np.pi/180,thresholds[c])
        if(len(lines) > 1):
            for i in range(len(lines)):
 

        c +=1
'''

# Find lines straight from edges
lines = cv2.HoughLines(edges,1,np.pi/180,110)


# Plot lines on image
posi = []
posf = []
for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        posi.append((x1,y1))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        posf.append((x2,y2))
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)


def near(a, b, rtol=1e-5, atol=1e-8):
    return abs(a - b) < (atol + rtol * abs(b))

# True if two lines cross, false if not
# should check that the three lines we found cross otherwise change threshold
def crosses(p1i, p1f, p2i, p2f):
    (x1, y1) = p1i
    (x2, y2) = p1f
    (v1, u1) = p2i
    (v2, u2) = p2f

    (a,b), (c,d) = (x2-x1, u1-u2), (y2-y1, v1-v2)
    e, f = u1-x1, v1-y1
    denom = float(a*d - b*c)
    if near(denom, 0):
        # parallel
        return False
    else:
        t = (e*d - b*f)/denom
        s = (a*f - e*c)/denom
        return 0<=t<=1 and 0<=s<=1


for i in range(len(posi)):
    for j in range(i+1, len(posi)):
        print(crosses(posi[i],posf[i],posi[j],posf[j]))


# Save image
cv2.imwrite('rectangles.jpg',img)


