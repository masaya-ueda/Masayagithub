import numpy as np
import cv2
img1 = cv2.imread(r'C:\Users\mueda\Documents\blog-thumb18.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create()
kp1 = surf.detect(gray1)
img1_surf = cv2.drawKeypoints(gray1, kp1, None, flags=4)
cv2.imshow('SURF', img1_surf)
cv2.waitKey()
cv2.imwrite('surf_mandy.jpg',img1_surf)