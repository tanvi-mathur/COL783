import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
def linear_transform(im):
    mat1=np.array(im, dtype=np.uint8)
    h2, w2 = im.shape[:2]
    hist = cv.calcHist([im],[0],None,[256],[0,256])
    h=[]
    for i in range(256):
        if hist[i]!=0:
            h.append(i)
    a=255/(max(h)-min(h))
    b=-a*min(h)
    for y in range(h2):
        for x in range(w2): 
            val=min(255, max(0, a*mat1[y][x]+b))
            mat1[y][x]=val
    return mat1

def linear_transform_manual(im, a=1, b=-2):
    mat1=np.array(im, dtype=np.float32)
    h2, w2 = im.shape[:2]
    for y in range(h2):
        for x in range(w2): 
            val=min(255, max(0, a*mat1[y][x]+b))
            mat1[y][x]=val
    return mat1
def median_threshold(im):
    mat1=np.array(im, dtype=np.float32)
    h2, w2 = im.shape[:2]
    sum1=0
    hist = cv.calcHist([im],[0],None,[256],[0,256])
    h=[]
    
    for i in range(256):
        if hist[i]!=0:
            h.append(i)
    h.sort()
    median=h[len(h)//2]
    mean=sum(h)//len(h)
    for y in range(h2):
        for x in range(w2):
            if mat1[y][x]>median:
                mat1[y][x]=255
            else:
                mat1[y][x]=0
    
    return mat1
def mean_threshold(im):
    mat1=np.array(im, dtype=np.float32)
    h2, w2 = im.shape[:2]
    sum1=0
    hist = cv.calcHist([im],[0],None,[256],[0,256])
    h=[]
    
    for i in range(256):
        if hist[i]!=0:
            h.append(i)
    h.sort()
    median=h[len(h)//2]
    mean=sum(h)//len(h)
    for y in range(h2):
        for x in range(w2):
            if mat1[y][x]>mean:
                mat1[y][x]=255
            else:
                mat1[y][x]=0
    
    return mat1

def hist_equal(im):
    mat1=np.array(im, dtype=np.uint8)
    h2, w2 = im.shape[:2]
    
    hist = cv.calcHist([im],[0],None,[256],[0,256])
    prob=[float(h[0]/(h2*w2)) for h in hist]
    cumu=[]
    i=0
    c=0
    while i < len(hist):
        c+=prob[i]
        i+=1
        cumu.append(c)
    memo={}

    # mat1=lut[im]
    for y in range(h2):
        for x in range(w2):
            val=mat1[y][x]
            if 255*cumu[val] not in memo:
                memo[val]=int(255*cumu[mat1[y][x]])
            mat1[y][x]=memo[val]    
    
    return mat1

new_image=linear_transform(mean_threshold(img))
cv.imshow("image", new_image)
k = cv.waitKey(0)

cv.imwrite("img_p1_2.png", new_image)