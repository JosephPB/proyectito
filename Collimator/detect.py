import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import peakutils
from PIL import Image

def detect(image):

    img = cv.imread(image)
    height,width,depth = img.shape

    #Averaging over intensities

    #Over columns

    tot_avg_width = np.array([])
    
    for i in range(width):
        avg = np.average(img[:,i])
        tot_avg_width = np.append(tot_avg_width,avg)
    

    x_width = np.arange(0,width)
    y_width = savgol_filter(tot_avg_width, window_length=101, polyorder=2, deriv=2)

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

    plt.plot(x1,line1, color = 'red')
    plt.plot(x1,line2, color = 'green')
    plt.plot(line3,x2, color = 'orange')
    plt.plot(line4,x2, color = 'blue')
    plt.imshow(img)
    plt.savefig('lines.jpg',dpi = 'figure')
    plt.show()


    #Crop image
    
    tl = [int(line4[0]),int(line1[0])]
    tr = [int(line3[0]),int(line1[0])]
    bl = [int(line4[0]),int(line2[0])]
    br = [int(line3[0]),int(line2[0])]

    x_corners = np.array([int(line4[0]),int(line3[0])])
    y_corners = np.array([int(line2[0]),int(line1[1])])

    crop = img[np.min(y_corners)+50:np.max(y_corners)-50,np.min(x_corners)+50:np.max(x_corners)-50]
    cv.imwrite('cropped.jpg',crop)
    
    plt.imshow(crop)
    #plt.show()

    #Calculate intensity of cropped image

    height,width,depth = crop.shape

    crop_avg_width = np.array([])
    
    for i in range(width):
        avg = np.average(img[:,i])
        crop_avg_width = np.append(crop_avg_width,avg)

    crop_intensity = np.sum(crop_avg_width)

    return crop_intensity



if __name__ == "__main__":

    origin_intensity = detect('wrist.jpg')

    
    #Try a rotation

    img2 = Image.open('wrist.jpg')

    rotated = img2.rotate(2)
    rotated.save('rotated.jpg')
    
#    rotated_intensity = detect('rotated.jpg')
#
#    if rotated_intensity > origin_intensity:
#        print('A rotation should be made')
#
#    else:
#        print('Rotating makes it worse')
    
