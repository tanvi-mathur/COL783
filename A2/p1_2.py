import cv2 as cv
import sys
import numpy as np
from scipy.signal import convolve2d
import time
import matplotlib.pyplot as plt


h = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
img=cv.imread(sys.argv[2], cv.IMREAD_GRAYSCALE)
psf=[10, 20, 50]
def bilinear_resize(im=h, pi=10):
    h,w=im.shape
    scale=pi/h
    
    new_img=np.array([[0 for i in range(int(w*scale))] for j in range(int(h*scale))], dtype=np.uint8)
   
    for y in range(int(h*scale)):
        for x in range(int(w*scale)):
            x_new, y_new=x/scale , y/scale 
            x1, y1= np.floor(x_new), np.floor(y_new)
            x2, y2=x1+1,y1+1
            dx,dy=x_new-x1, y_new-y1
            
            # new_img[y][x]=old_img[min(y_new+int(min(new_points_y)), h-1)][min(x_new+int(min(new_points_x)), w-1)]
            val=min(255, im[min(h-1, int(y1))][min(w-1, int(x1))]*(1-dx)*(1-dy)+im[min(h-1, int(y2))][min(w-1, int(x1))]*dy*(1-dx)+im[min(h-1, int(y1))][min(w-1, int(x2))]*dx*(1-dy)+im[min(h-1, int(y2))][min(w-1, int(x2))]*dx*dy)
            new_img[y][x] = val   
      

    return new_img
def fourier_transform_filter(im, pi):
    
    h,w=im.shape
    F = np.fft.fft2(im)    
    F = np.fft.fftshift(F)
    # F=cv.dft(np.float32(im), flags=cv.DFT_COMPLEX_OUTPUT)
    H=np.zeros((h,w))
    k=pi/h
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

def convolve(kernel, im):
    
    b, a=kernel.shape
    h, w=im.shape
    
    b1, a1 = (b)//2, (a)//2
    new_img=np.zeros((h, w))
    # pimg = np.zeros((h+b-1, w+a-1), dtype=np.float32)
    # pimg[b1:b1+h, a1:a1+w] = im  

    # pimg[0:b1, a1:a1+w] = im[b1:0:-1, :]        
    # pimg[b1+h:, a1:a1+w] = im[-2:-b1-2:-1, :]  

    # pimg[:, 0:a1] = pimg[:, 2*a1:a1:-1]         
    # pimg[:, a1+w:] = pimg[:, w+a1-2:w-2:-1]
    kernel = np.flipud(np.fliplr(kernel))
    pimg = np.pad(im,pad_width=((b1, b1), (a1, a1)),mode='reflect')
    # for y in range(b1, h+b1+1):
    #     for x in range(a1, w+a1+1):
            
    #         # for t in range(-b1, b1+1):
    #         #     for s in range(-a1, a1+1):
    #         #         val+=kernel[t+b1][s+a1]*pimg[y-t][x-s]
    #         window = pimg[y+b1+1:y-b1:-1, x+a1:x-a1:-1]
    #         new_img[y-b1][x-a1]=np.sum(window*kernel)
    for y in range(h):
        for x in range(w):
            window = pimg[y:y+b, x:x+a]  
            new_img[y, x] = np.sum(window * kernel)
    return new_img
def fourier_mult(h, im):
    b, a=h.shape
    h1,w1=im.shape
    b1, a1 = (b-1)//2, (a-1)//2
    pimg = np.pad(im,pad_width=((b1, b1), (a1, a1)),mode='reflect')

    F_im = np.fft.fft2(pimg)    

    H = np.zeros(pimg.shape)
    kh, kw = h.shape
    H[:kh, :kw] = h
    Hf = np.fft.fft2(H)
    G = F_im*Hf
    blurred_freq = np.real(np.fft.ifft2(G)[b1:h1+b1, a1:a1+w1])
    blurred = cv.normalize(blurred_freq, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)

    return blurred

F_h=np.fft.fft2(h)
F_im=np.fft.fft2(img)

ver_psf=[]
for p in psf:
    im_filtered=fourier_transform_filter(h, p)
    new_image=bilinear_resize(im_filtered, p)
    ver_psf.append(new_image)
psf_versions = [p.astype(np.float32) / np.sum(p) for p in ver_psf]
blurred_versions = []
blurred_versions_F=[]
t_s=[]
t_f=[]
for i, p in enumerate(psf_versions):
    start_time_1= time.perf_counter()
    blurred_F=fourier_mult(p,img)
    end_time_1=time.perf_counter()
    t_f.append(end_time_1-start_time_1)
    start_time_2 = time.perf_counter()
    blurred = convolve(p, img)
    end_time_2=time.perf_counter()
    t_s.append(end_time_2-start_time_2)
    
    blurred_versions.append(blurred)
    blurred_versions_F.append(blurred_F)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite(f"/orig_img_psf_{i+1}.png", blurred)
    cv.imwrite(f"/orig_img_psf_fourier_mult_{i+1}.png", blurred_F)
plt.title("In Fourier domain")
plt.plot(psf, t_f)
plt.xlabel('Kernel size')
plt.ylabel('Time')
plt.show()
plt.title("In Spatial domain")
plt.plot(psf, t_s)
plt.xlabel('Kernel size')
plt.ylabel('Time')
plt.show()
