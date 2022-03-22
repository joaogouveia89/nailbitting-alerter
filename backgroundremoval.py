import numpy as np
import cv2


frame = cv2.imread("report/nb.jpg")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# try this in order to preprocess the image
ret,divided = cv2.threshold(gray,127,255,cv2.THRESH_BINARY) 


cv2.imshow("Grayscale", gray)
cv2.imshow("Divided", divided)

print(gray)
print(divided)

cv2.waitKey(0) 

cv2.destroyAllWindows() 