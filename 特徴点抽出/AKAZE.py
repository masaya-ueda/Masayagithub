import numpy as np
import cv2
img1 = cv2.imread(r'C:\Users\mueda\Documents\blog-thumb18.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

detector = cv2.AKAZE_create()
kp1 = detector.detect(gray1)
img1_detector = cv2.drawKeypoints(gray1, kp1, None, flags=4)
cv2.imshow('AKAZE', img1_detector)
cv2.waitKey()
cv2.imwrite('akaze_mandy.jpg',img1_detector)