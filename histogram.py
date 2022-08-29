import cv2
import numpy as np
import time
import sys
import os.path
import keyboard
# Python gradient calculation 

# Read image
img = cv2.imread(r"C:\Users\mattp\Documents\thesis_Stuff\images\original\image_0022.jpg")
img = np.float32(img) / 255.0
block = img[1200:1200+8,1000:1000+8]
block = np.float32(block) / 255.0
# Calculate gradient
gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)


#block_mag, block_angle = cv2.cartToPolar(block_gx, block_gy, angleInDegrees=True)
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

combined = cv2.addWeighted(gx,0.5,gy,0.5,0)


#block_combined = cv2.addWeighted(block_gx,0.5,block_gy,0.5,0)
cv2.namedWindow("histo_test.jpg", cv2.WINDOW_NORMAL)

cv2.imshow("histo_test.jpg", combined)
cv2.waitKey(0)
print(combined[1])
#print(block_angle)