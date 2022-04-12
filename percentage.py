import cv2
import numpy as np

img1 = cv2.imread('trainning_data/nb/10.jpg')
img2 = cv2.imread('trainning_data/nb/12.jpg')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

img1 = cv2.GaussianBlur(img1,(9,9),500)
img2 = cv2.GaussianBlur(img2,(9,9),500)

#--- take the absolute difference of the images ---
res = cv2.absdiff(img1, img2)

cv2.imshow('diff',res)

#--- convert the result to integer type ---
res = res.astype(np.uint8)

#--- find percentage difference based on number of pixels that are not zero ---
percentage = (np.count_nonzero(res) * 100)/ res.size

print(percentage)

cv2.waitKey(0)
cv2.destroyAllWindows()