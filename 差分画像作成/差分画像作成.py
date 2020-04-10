import numpy as np
import cv2
img1 = cv2.imread(r'C:\Users\mueda\Documents\blog-thumb18.jpg',0)
img2 = cv2.imread(r'C:\Users\mueda\Documents\blog-thumb18_02.jpg',0)
bgObj = cv2.bgsegm.createBackgroundSubtractorMOG()
fgmask = bgObj.apply(img1)
fgmask = bgObj.apply(img2)
 
cv2.imshow('frame', fgmask)
cv2.waitKey()