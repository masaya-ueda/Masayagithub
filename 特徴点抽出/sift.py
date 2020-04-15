import numpy as np
import cv2
img1 = cv2.imread(r'C:\Users\mueda\Documents\blog-thumb18.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp1 = sift.detect(gray1)
img1_sift = cv2.drawKeypoints(gray1, kp1, None, flags=4)
cv2.imshow('SIFT', img1_sift)
cv2.waitKey()
cv2.imwrite('sift_mandy.jpg',img1_sift)