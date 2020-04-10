import numpy as np
import cv2
img = cv2.imread(r'C:\Users\mueda\Documents\blog-thumb18.jpg')
height = img.shape[0]
width = img.shape[1]
img2 = cv2.resize(img , (int(width*0.5), int(height*0.5)))
cv2.imshow('image',img2)
cv2.waitKey()