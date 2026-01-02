import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt
import time
img = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)


def convolve(kernel, im, c=0):
    mat1=np.array(im, dtype=np.float32)
    
    b, a=kernel.shape
    h, w=im.shape
    
    b1, a1 = (b-1)//2, (a-1)//2
   
    new_img=np.zeros((h, w))
    pimg = np.zeros((h+b-1, w+a-1), dtype=np.float32)
    pimg[b1:h+b1, a1:w+a1] = im
    for j in range(h):
        pimg[j+b1, 0:a1]=im[j,0]
        pimg[j+b1, w+a1:w+a]=im[j, w-1]
    for i in range(w):
        pimg[0:b1, i+a1]=im[0, i]
        pimg[h+b1:h+b, i+a1]=im[h-1, i]
    
    for y in range(b1, h+b1):
        for x in range(a1, w+a1):
            val=0
            for t in range(-b1, b1+1):
                for s in range(-a1, a1+1):
                    val+=kernel[t+b1][s+a1]*pimg[y-t][x-s]
            v=val+c
            new_img[y-b1][x-a1]=v
    return new_img


def laplace(im, c, alpha=0.2):
    mat1=np.array(im, dtype=np.float32)
    
    h2, w2 = im.shape[:2]
    out=np.zeros((h2, w2))
    kernel=np.array([[0, -1, 0], [-1,4,-1], [0,-1,0]])
    # for m in range(1, len(mat1)-1):
    #     for n in range(1, len(mat1[0])-1):
    #         v=0
    #         for s in range(len(kernel)):
    #             for t in range(len(kernel[0])):
    #                 v+=mat1[m+s-1][n+t-1]*kernel[s][t]
    #         val=max(0, min(255, v))
    #         mat[m][n]=max(0, min(mat1[m][n]+v, 255))
    mat=convolve(kernel, im, c)
    out = np.clip(mat1 + alpha * mat, 0, 255)
    
    return mat

def laplace_kernel():
    kernel=np.array([[0, -1, 0], [-1,4,-1], [0,-1,0]])
    return kernel

def gaussian_kernel(n1=3, sigma=4):
   
    kernel=np.zeros((n1, n1))
    e=2.71828
    sum1=0
    memo={}
    for j in range(-(n1-1)//2, (n1-1)//2+1):
        l=[]        
        for i in range(-(n1-1)//2, (n1-1)//2+1):
            if (i**2+j**2) not in memo.keys():                 
                memo[i**2+j**2]=e**(-(i**2+j**2)/(2*(sigma)**2))
            kernel[j+(n1-1)//2, i+(n1-1)//2]=memo[i**2+j**2]          
            sum1+= memo[i**2+j**2]  
    kernel/=sum1
    return kernel

def gaussian(im, n1=10, sigma=4, c=0):
    mat1=np.array(im, dtype=np.uint8)
    start_time = time.perf_counter()
    # mean=0
    # square=0
    # h2, w2 = im.shape[:2]
    # for m in range(len(mat1)):
    #     for n in range(len(mat1[0])):
    #         mean+=mat1[m][n]
    #         square+=mat1[m][n] ** 2
    # mean=mean/(h2*w2)
    # square=square/(h2*w2)
    # var=square-(mean**2)
    kernel=np.zeros((n1, n1))
    e=2.71828
    sum1=0
    memo={}
    for j in range(-(n1-1)//2, (n1-1)//2+1):
        l=[]        
        for i in range(-(n1-1)//2, (n1-1)//2+1):
            if (i**2+j**2) not in memo.keys():                 
                memo[i**2+j**2]=e**(-(i**2+j**2)/(2*(sigma)**2))
            kernel[j+(n1-1)//2, i+(n1-1)//2]=memo[i**2+j**2]          
            sum1+= memo[i**2+j**2]  
    kernel/=sum1
    # for m in range(len(mat1)-n1+1):
    #     for n in range(len(mat1[0])-n1+1):
    #         v=0
    #         for s in range(n1):
    #             for t in range(n1):
    #                 v+=mat1[m+s-1][n+t-1]*kernel[s][t]
    #         v/=sum1
    #         mat[m][n]=max(0, min(255, v))
    mat= convolve(kernel, im, c)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"Elapsed time using Gaussian 2D kernel: {elapsed_time:.6f} seconds")
    return mat

def gaussian_sep(im, n1=10, sigma=4, c=0):
    mat1=np.array(im, dtype=np.float32)
    start_time = time.perf_counter()
    memo={}
    # mean=0
    # square=0
    # h2, w2 = im.shape[:2]
    # for m in range(len(mat1)):
    #     for n in range(len(mat1[0])):
    #         mean+=mat1[m][n]
    #         square+=mat1[m][n] ** 2
    # mean=mean/(h2*w2)
    # square=square/(h2*w2)
    # var=square-(mean**2)
    kernel_x=np.zeros((1, n1))
    kernel_y=np.zeros((n1, 1))
    e=2.71828
    sum_x=0 
   
    for i in range(-(n1-1)//2, (n1-1)//2+1):  
        val_x=e**(-i**2/(2*(sigma)**2))
        memo[i]=val_x      
        kernel_x[0, i+(n1-1)//2]=val_x    
        sum_x+=val_x
    for j in range(-(n1-1)//2, (n1-1)//2+1):       
        kernel_y[j+(n1-1)//2, 0]=memo[j]     
        
    kernel_x/=sum_x
    kernel_y/=sum_x
    # for m in range(len(mat1)-n1+1):
    #     for n in range(len(mat1[0])-n1+1):
    #         v=0
    #         for s in range(n1):
    #             for t in range(n1):
    #                 v+=mat1[m+s-1][n+t-1]*kernel[s][t]
    #         v/=sum1
    #         mat[m][n]=max(0, min(255, v))
    mat= convolve(kernel_x, im)
    mat2=convolve(kernel_y, mat, c)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time using Gaussian 1D kernel: {elapsed_time:.6f} seconds")
    return mat2
def log_kernel(im, n1=2, sigma=2, c=128):
    memo={}
    
    kernel_x=np.zeros((1, n1))
    kernel_y=np.zeros((n1, 1))
    e=2.71828
    sum_x=0 
   
    for i in range(-(n1-1)//2, (n1-1)//2+1):  
        val_x=e**(-i**2/(2*(sigma)**2))
        memo[i]=val_x      
        kernel_x[0, i+(n1-1)//2]=val_x    
        sum_x+=val_x
    for j in range(-(n1-1)//2, (n1-1)//2+1):       
        kernel_y[j+(n1-1)//2, 0]=memo[j]     
        
    kernel_x/=sum_x
    kernel_y/=sum_x
    new_kernel_x=convolve(kernel_x,laplace_kernel(), 0)
    new_kernel_y=convolve(kernel_y, laplace_kernel(),  0)
    
    mat= convolve(new_kernel_x, im, c)
    mat1=convolve(new_kernel_y, mat, c)
    return mat1
new_image_lg=laplace(gaussian_sep(img, 3, 3), 128)
new_image_gl=gaussian_sep(laplace(img, 0), 3, 3, 128)
new_image_log=np.clip(log_kernel(img), 0, 255)
# new_image=laplace(img, 128)
cv.imshow("image", new_image_log)

k = cv.waitKey(0)

cv.imwrite("p2_1_log.png", new_image_log)
cv.imwrite("p2_1_lg.png", new_image_lg)
cv.imwrite("p2_1_gl.png", new_image_gl)
