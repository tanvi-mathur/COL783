import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
def select_corners(im):
    scale_percent = 20 
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)  
    cv.imshow("image", resized)
    points = []
    def click_event(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Point selected: {(x, y)}")
            cv.circle(resized, (x, y), 5, (0, 255, 0), -1)
            cv.imshow("image", resized)    
    cv.setMouseCallback("image", click_event)
    cv.waitKey(0)
    cv.destroyAllWindows()
    points=[[p[0]*100/scale_percent, p[1]*100/scale_percent] for p in points]
    print("Selected points:", points)
    return points
points=select_corners(img)


def rotation_nearest_neighbour(im):
    tantheta = (points[0][1]-points[1][1])/(points[1][0]-points[0][0])
    costheta=(1+tantheta**2)**(-0.5)
    sintheta=(1+(tantheta+1e-6)**(-2))**(-0.5) * (tantheta+1e-6)/abs(tantheta+1e-6)
    old_img=np.array(im, dtype=np.uint8)
    h, w=im.shape
    #new_points_y = [points[0][1]]+[-(p[0]-points[0][0])*sintheta+(p[1]-points[0][1])*costheta + points[0][1] for p in points[0:]]
    #new_points_x = [points[0][0]]+[(p[0]-points[0][0])*costheta+(p[1]-points[0][1])*sintheta + points[0][0] for p in points]
    points_y=[p[1] for p in points]
    points_x=[p[0] for p in points]
    # h1=int(max(new_points_y))-int(min(new_points_y))
    # w1=int(max(new_points_x))-int(min(new_points_x))
    h1=int(abs(points[0][1]-points[-1][1]))
    w1=int(abs(points[0][0]-points[1][0]))
    new_img=np.array([[0 for j in range(w1)] for i in range(h1)], dtype=np.uint8)
    
    for y in range(h1):
        for x in range(w1):            
            y_new=int(-x*sintheta+y*costheta+max(points_y[0], points_y[1]))             
            x_new=int(y*sintheta+x*costheta+min(points_x[0], points_x[-1]))
            # new_img[y][x]=old_img[min(y_new+int(min(new_points_y)), h-1)][min(x_new+int(min(new_points_x)), w-1)]
            val=old_img[min(h-1, int(y_new))][min(w-1, int(x_new))]
            new_img[y][x] = val

    return new_img

def rotation_bilinear(im):
    tantheta = (points[0][1]-points[1][1])/(points[1][0]-points[0][0])
    costheta=(1+tantheta**2)**(-0.5)
    sintheta=(1+(tantheta+1e-6)**(-2))**(-0.5) * (tantheta+1e-6)/abs(tantheta+1e-6)
    old_img=np.array(im, dtype=np.uint8)
    h,w=im.shape
    
    #new_points_y = [points[0][1]]+[-(p[0]-points[0][0])*sintheta+(p[1]-points[0][1])*costheta + points[0][1] for p in points[0:]]
    #new_points_x = [points[0][0]]+[(p[0]-points[0][0])*costheta+(p[1]-points[0][1])*sintheta + points[0][0] for p in points]
    points_y=[p[1] for p in points]
    points_x=[p[0] for p in points]
    # h1=int(max(new_points_y))-int(min(new_points_y))
    # w1=int(max(new_points_x))-int(min(new_points_x))
    h1=int(abs(points[0][1]-points[-1][1]))
    w1=int(abs(points[0][0]-points[1][0]))
    new_img=np.array([[0 for j in range(w1)] for i in range(h1)], dtype=np.uint8)
    
    for y in range(h1):
        for x in range(w1):            
            y_new=(-x*sintheta+y*costheta)+max(points_y[0], points_y[1])     
            x_new=(y*sintheta+x*costheta)+min(points_x[0], points_x[-1])
            x1, y1= np.floor(x_new), np.floor(y_new)
            x2, y2=x1+1,y1+1
            dx,dy=x_new-x1, y_new-y1
            # new_img[y][x]=old_img[min(y_new+int(min(new_points_y)), h-1)][min(x_new+int(min(new_points_x)), w-1)]
            val=old_img[min(h-1, int(y1))][min(w-1, int(x1))]*(1-dx)*(1-dy)+old_img[min(h-1, int(y2))][min(w-1, int(x1))]*dy*(1-dx)+old_img[min(h-1, int(y1))][min(w-1, int(x2))]*dx*(1-dy)+old_img[min(h-1, int(y2))][min(w-1, int(x2))]*dx*dy
            new_img[y][x] = val
    return new_img

new_image_nearest_neighbour=rotation_nearest_neighbour(img)
new_image_bilinear_interpolation=rotation_bilinear(img)

k = cv.waitKey(0)

cv.imwrite("img_rot_nearest_neighbour.png", new_image_nearest_neighbour)
cv.imwrite("img_rot_nearest_neighbour.png", new_image_bilinear_interpolation)
