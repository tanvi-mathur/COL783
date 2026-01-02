import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread(sys.argv[2], cv.IMREAD_GRAYSCALE)
logo=cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)

def mean_threshold(im):
    mat1=np.array(im, dtype=np.float32)
    h2, w2 = im.shape[:2]
    hist = cv.calcHist([im],[0],None,[256],[0,256])
    h=[]
    
    for i in range(256):
        if hist[i]!=0:
            h.append(i)
    h.sort()
    mean=sum(h)//len(h)
    for y in range(h2):
        for x in range(w2):
            if mat1[y][x]>mean:
                mat1[y][x]=0
            else:
                mat1[y][x]=1
    
    return mat1
mask=mean_threshold(logo)
def bilinear_grayscale(im=logo, scale=0.2):
    
    mat1=list(np.array(im, dtype=np.uint8))
    h,w=im.shape
    pimg = np.zeros((h+4, w+4))
    pimg[1:h+1, 1:w+1] = im
 
    pimg[2:h+2, 0:2] = im[:, 0:1]
    pimg[h+2:h+4, 2:w+2] = im[h-1:h, :]
    pimg[2:h+2, w+2:w+4] = im[:, w-1:w]
    pimg[0:2, 2:w+2] = im[0:1, :]
    
    pimg[0:2, 0:2] = im[0, 0]
    pimg[h+2:h+4, 0:2] = im[h-1, 0]
    pimg[h+2:h+4, w+2:w+4] = im[h-1, w-1]
    pimg[0:2, w+2:w+4] = im[0, w-1]
    new_img=np.array([[0 for i in range(int(w*scale))] for j in range(int(h*scale))], dtype=np.uint8)
    out=[]
    for y in range(int(h*scale)):
        for x in range(int(w*scale)):
            x_new, y_new=x/scale + 2, y/scale + 2
            x1, y1= np.floor(x_new), np.floor(y_new)
            x2, y2=x1+1,y1+1
            dx,dy=x_new-x1, y_new-y1
            
            # new_img[y][x]=old_img[min(y_new+int(min(new_points_y)), h-1)][min(x_new+int(min(new_points_x)), w-1)]
            val=min(255, pimg[min(h+3, int(y1))][min(w+3, int(x1))]*(1-dx)*(1-dy)+pimg[min(h+3, int(y2))][min(w+3, int(x1))]*dy*(1-dx)+pimg[min(h+3, int(y1))][min(w+3, int(x2))]*dx*(1-dy)+pimg[min(h+3, int(y2))][min(w+3, int(x2))]*dx*dy)
            new_img[y][x] = val   

    return new_img

# cv.imshow("image", bilinear_grayscale(mask))
cv.waitKey(0)          

def insert(im1=logo, im2=img, a=0.2):
    mat1=np.array(im1, dtype=np.uint8)
    mat2=np.array(im2, dtype=np.uint8)
    h1, w1=im1.shape
    h2,w2=im2.shape
    alpha = a*w2/w1
    logo_resize=bilinear_grayscale(im1, alpha)
    mask=mean_threshold(logo_resize)
    h3, w3=logo_resize.shape
    
    for y in range(h3):
        for x in range(w3):
            val=min(255, max(0, (1-mask[y][x])*mat2[min(h2-1, int(y+h2*(1-a)))][min(w2-1, int(x+w2*(1-a)))]+mask[y][x]*(0.5*mat2[min(h2-1, int(y+h2*(1-a)))][min(w2-1, int(x+w2*(1-a)))]+0.5*logo_resize[y][x])))               
            mat2[min(h2-1, int(y+h2*(1-a)))][min(w2-1, int(x+w2*(1-a)))]=val
    return mat2

new_image=insert(logo, img)
cv.imshow("image", new_image)
k = cv.waitKey(0)

cv.imwrite("img_p1_3.png", new_image)