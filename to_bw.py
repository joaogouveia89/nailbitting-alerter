import cv2
from os import listdir
from os import remove
from os.path import isfile, join

onlyfiles = [f for f in listdir("trainning_data/nnb") if isfile(join("trainning_data/nnb", f))]

for img in onlyfiles:
    originalImage = cv2.imread('trainning_data/nnb' + "/" + img)
    remove('trainning_data/nnb' + "/" + img)
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('trainning_data/nnb' + "/" + img, grayImage)


