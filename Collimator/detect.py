import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import peakutils
from PIL import Image
import sys

def dark(image):
    '''
    Checks image borders for dark pixels (with an RGB value of less than 20).
    Calculates the ratio of dark pixels to total pixels in border region.
    Input: Image path
    Output: Boolean operator for collimation
    '''

    img = cv.imread(image)
    cp = np.copy(img)
    cp[cp<100] = 1
    cp[cp>150] = 255

    height,width,depth = cp.shape
    num_dark_left = np.count_nonzero(cp[:,0:width/10] < [20])/3
    num_dark_right = np.count_nonzero(cp[:,-width/10:width] < [20])/3
    num_dark_top = np.count_nonzero(cp[0:height/10,:] < [20])/3
    num_dark_bottom = np.count_nonzero(cp[-height/10:height,:] < [20])/3
    num_dark = num_dark_left+num_dark_right+num_dark_bottom+num_dark_top
    
    total = (width/10)*height*2 + (height/10)*width*2
    
    ratio = (float(num_dark)/float(total))

    if ratio > 0.6:
        print ('image collimated')
        return True
    
    else: 
        print False

    
def detect(image, write, plot, rotation):
    '''
    If a collimator is identified, crops image down.
    Input:
     - image: input image path
     - write: write out image path
     - plot: bool, plot intermediate results (True)
     - rotation: bool, implement a random rotation on input image
    '''
    
    if rotation == True:
        img = Image.open(image)
        rotation = np.random.random()*89
        img = img.rotate(rotation)
        img.save('rotated.jpg')
        img = cv.imread('rotated.jpg')

        if plot == True:
            plt.imshow(img)
            plt.show()

    else:
        img = cv.imread(image)

    img = img[400:2400,500:2700] #Crop down artificial border
    height,width,depth = img.shape

    #Detect rotation
    edges = cv.Canny(img, 0, 30, 3)
    cp2 = np.copy(img) #Create a copy

    lines = cv.HoughLines(edges,50,np.pi/180,1)
    rho, theta = lines[0][0]

    gradient = np.sin(theta)/np.cos(theta)
    x = np.linspace(0,-width,width)
    y = gradient*x

    if plot == True:
        plt.imshow(cp2)
        plt.plot(x,y)
        plt.show()

    #Rotate image back    
    angle = theta*(180/np.pi)
    img = Image.open(image)
    img = img.rotate(rotation)
    img = img.rotate(angle+180)

    img.save('rotated.jpg')
    img = cv.imread('rotated.jpg')
    img = img[400:2400,500:2700]

    if plot == True:
        plt.imshow(img)
        plt.show()
    
    
    #Averaging over intensities

    #Over columns

    tot_avg_width = np.array([])
    
    for i in range(width):
        avg = np.average(img[:,i])
        tot_avg_width = np.append(tot_avg_width,avg)
    

    x_width = np.arange(0,width)
    y_width = savgol_filter(tot_avg_width, window_length=101, polyorder=2, deriv=2)

    if plot == True:
        plt.plot(x_width,y_width)
        plt.title('Second derivatives of average intensities by column')
        plt.savefig('second_column.jpg',dpi = 300)
        plt.show()

    #Over rows

    tot_avg_height = np.array([])
    
    for i in range(height):
        avger = np.average(img[i,:])
        tot_avg_height = np.append(tot_avg_height,avger)

        
    x_height = np.arange(0,height)
    y_height = savgol_filter(tot_avg_height, window_length=101, polyorder=2, deriv=2)


    #Peak detection
    
    peakfind_width = peakutils.indexes(y_width, thres=0.1, min_dist=500)
    peakfind_height = peakutils.indexes(y_height, thres=0.1, min_dist=500)

    peaks_width = y_width[peakfind_width]
    peaks_height = y_height[peakfind_height]

    sorted_width = np.argsort(peaks_width)
    two_max_width = sorted_width[::-1][:2]

    sorted_height = np.argsort(peaks_height)
    two_max_height = sorted_height[::-1][:2]

    #Vertical lines
    line3 = np.linspace(peakfind_width[two_max_width[0]],peakfind_width[two_max_width[0]],height)
    line4 = np.linspace(peakfind_width[two_max_width[1]],peakfind_width[two_max_width[1]],height)

    x2 = np.linspace(0,height,height)

    #Horizontal lines
    line1 = np.linspace(peakfind_height[two_max_height[0]],peakfind_height[two_max_height[0]],width)
    line2 = np.linspace(peakfind_height[two_max_height[1]],peakfind_height[two_max_height[1]],width)

    x1 = np.linspace(0,width,width)

    if plot == True:
        plt.plot(x1,line1, color = 'red')
        plt.plot(x1,line2, color = 'green')
        plt.plot(line3,x2, color = 'orange')
        plt.plot(line4,x2, color = 'blue')
        plt.imshow(img)
        plt.savefig('lines.jpg',dpi = 300)
        plt.show()


    #Crop image
    
    tl = [int(line4[0]),int(line1[0])]
    tr = [int(line3[0]),int(line1[0])]
    bl = [int(line4[0]),int(line2[0])]
    br = [int(line3[0]),int(line2[0])]

    x_corners = np.array([int(line4[0]),int(line3[0])])
    y_corners = np.array([int(line2[0]),int(line1[1])])

    crop = img[np.min(y_corners)+50:np.max(y_corners)-50,np.min(x_corners)+50:np.max(x_corners)-50]
    cv.imwrite(write + 'cropped.jpg',crop)

    if plot == True:
        plt.imshow(crop)



if __name__ == "__main__":

#    detect('/media/sf_TestImageDataBase/rectangle/jpeg_converted/NeckofFemur.jpg',True,False)

    collimated = dark(sys.argv[1])

    if collimated == True:
        detect(sys.argv[1],sys.argv[2],False,True)
