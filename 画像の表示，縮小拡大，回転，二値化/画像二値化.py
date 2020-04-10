import numpy as np
import cv2
img = cv2.imread(r'C:\Users\mueda\Documents\blog-thumb18.jpg',0)
ret2, img_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

print("ret2: {}".format(ret2))

cv2.imshow('image',img_otsu)
cv2.waitKey()