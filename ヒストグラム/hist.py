import cv2  
import matplotlib.pyplot as plt 

fname_1 = r'C:\Users\mueda\Documents\blog-thumb40.jpg' 
fname_2 = r'C:\Users\mueda\Documents\blog-thumb18.jpg' 

img_1 = cv2.imread(fname_1) 
img_2 = cv2.imread(fname_2)

hist_g_1 = cv2.calcHist([img_1],[2],None,[256],[0,256]) 
plt.plot(hist_g_1,color = "r") 
plt.show() 

hist_g_2 = cv2.calcHist([img_2],[2],None,[256],[0,256]) 
plt.plot(hist_g_2,color = "r")
plt.show() 

comp_hist = cv2.compareHist(hist_g_1, hist_g_2, cv2.HISTCMP_CORREL) 
print(comp_hist) 