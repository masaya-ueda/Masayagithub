import numpy as np
import cv2
img = cv2.imread(r'C:\Users\mueda\Documents\blog-thumb18.jpg')
height = img.shape[0]
width = img.shape[1]
center = (int(width/2), int(height/2))
angle = 90
scale = 1
trans = cv2.getRotationMatrix2D(center, angle , scale)
img2 = cv2.warpAffine(img, trans, (width,height))
cv2.imshow('image',img2)
cv2.waitKey()