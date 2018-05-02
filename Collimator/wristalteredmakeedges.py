import cv2
import matplotlib.pyplot as plt
import numpy as np



def condition(xs,ys):
    if(len(ys))==0:
        return True
    else:
        for i in range(len(ys)):
            if(abs(xs[0] - ys[i][0]) <= 800 and  abs(xs[1] - ys[i][1]) <= 8):
                return False
    return True
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
        #return 0<=t<=1 and 0<=s<=1
        return t,s
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


im = cv2.imread('wristaltered.jpg')
edges = cv2.Canny(im,0,50,apertureSize = 3)
cv2.imwrite('wristalterededge.jpg',edges)


lines = cv2.HoughLines(edges, 1, np.pi/180,200)


# Plot lines on image
posi = []
posf = []
valid_lines = []

print(' Number of lines found : ', len(lines))

if 0==0:

    for line in lines:
        if condition(line[0],valid_lines):
            valid_lines.append(line[0].tolist())

    print('Number of distinct lines found : ',len(valid_lines))

    for line in valid_lines:
        rho = line[0]
        theta = line[1]
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
        cv2.line(im,(x1,y1),(x2,y2),(0,0,255),8)


   # Save image
    pxs = []
    pys = []
    for i in range(len(posi)):
        for j in range(i+1, len(posi)):
            px,py = line_intersection([posi[i],posf[i]],[posi[j],posf[j]])
            pxs.append(int(px))
            pys.append(int(py))
            #cv2.circle(im,(int(px),int(py)),300, (0,255,0),2 )
    
    cv2.rectangle(im,(pxs[0],pys[0]),(pxs[1],pys[1]),(0,255,0),3)
    plt.imshow(im)
    plt.show()
    cv2.imwrite('wristalteredrec.jpg',im)


else:
    print ("length of lines is 0")
