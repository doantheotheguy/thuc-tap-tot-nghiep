import numpy as np
import cv2
import matplotlib.pylab as plt
img = cv2.imread('tttn4.jpg', 1)
cv2.imshow('Org img', img)
    #grey_img = cv2.imread('tttn2.jpg', 0)
    #darkened_img = cv2.addWeighted(grey_img,.7,grey_img,0,0) ##cần tăng độ tương phản
    #hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    #yellowmask = cv2.inRange(hls_img,np.array([0,100,100]),np.array([50,250,250])) #cần chỉnh để lấy màu vàng tttn2
    #whitemask = cv2.inRange(darkened_img,150,255)
    #commonmask = cv2.bitwise_or(yellowmask, whitemask)
    #ready2handle = cv2.bitwise_and(darkened_img, commonmask)
    #gaussblur = cv2.GaussianBlur(ready2handle,(5,5),0)
    #edges = cv2.Canny(ready2handle,100,200)
    #lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    #for line in lines:
#    x1,y1,x2,y2 = line[0]
#       cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    #cv2.imshow('Darkened img', darkened_img)
    #cv2.imshow('HLS img', hls_img)
    #cv2.imshow('yellowmask', yellowmask)
    #cv2.imshow('whitemask', whitemask)
    #cv2.imshow('commonmask', commonmask)
    #cv2.imshow('ready2handle', ready2handle)
    #cv2.imshow('gaussblur', gaussblur)
#   cv2.imshow('edge', edges)
#   cv2.imshow('lines', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
