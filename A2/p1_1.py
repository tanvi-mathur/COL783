import cv2 as cv
import sys
import numpy as np

img = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
k=0.2
def bilinear_resize(im=img, scale=k):
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
def nearest_neighbour_resize(im=img, scale=0.5):
    
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
            
            # new_img[y][x]=old_img[min(y_new+int(min(new_points_y)), h-1)][min(x_new+int(min(new_points_x)), w-1)]
            val=min(255, pimg[min(h+3, int(y_new)), min(w+3, int(x_new))])
            new_img[y][x] = val   

    return new_img

def fourier_transform_filter(im, k):
    h,w=im.shape
    F = np.fft.fft2(im)    
    F = np.fft.fftshift(F)
    # F=cv.dft(np.float32(im), flags=cv.DFT_COMPLEX_OUTPUT)
    H=np.zeros((h,w))
    dx=1
    dy=1
    f_x_nyq=k/(2*dx)
    f_y_nyq=k/(2*dy)
    F_new=np.zeros(F.shape)
    for nu_y in range(-h//2, h//2):
        for nu_x in range(-w//2, w//2):
            # elliptical
            if (nu_y/(h*f_y_nyq))**2+(nu_x/(w*f_x_nyq))**2<=1:
                H[nu_y+h//2, nu_x+w//2]=1
    F_new=F*H
    
    im_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(F_new)))
    magnitude = cv.normalize(im_filtered, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
    return magnitude

im_filtered=fourier_transform_filter(img, k)
new_image_bilinear_after_fft=bilinear_resize(im_filtered, k)
new_image_bilinear=bilinear_resize(img, k)
bilinear_fft=cv.dft(np.float32(new_image_bilinear), flags=cv.DFT_COMPLEX_OUTPUT)
fourier_shift = np.fft.fftshift(bilinear_fft)
magnitude = 20*np.log(cv.magnitude(fourier_shift[:,:,0],fourier_shift[:,:,1]))
magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
cv.imshow('Fourier Transform', im_filtered)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite("img_bilinear_fft.png", magnitude)
cv.imwrite("img_bilinear_before_fft.png", new_image_bilinear)
cv.imwrite("img_bilinear_after_fft.png", new_image_bilinear_after_fft)
cv.imwrite("orig_img_bilinear_before_fft.png", nearest_neighbour_resize(new_image_bilinear, 1/k))
cv.imwrite("orig_img_bilinear_after_fft.png", nearest_neighbour_resize(new_image_bilinear_after_fft, 1/k))
