import cv2
img1 = cv2.imread(r'C:\Users\mueda\Documents\S__41476104.jpg')
img2 = cv2.imread(r'C:\Users\mueda\Documents\S__41476106.jpg')
sift = cv2.xfeatures2d.SIFT_create()                                
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
ratio = 0.5
good = []
for m, n in matches:
    if m.distance < ratio * n.distance:
        good.append([m])
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:100], None, flags=2)
cv2.imshow('img', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('sift_matching.jpg',img3)