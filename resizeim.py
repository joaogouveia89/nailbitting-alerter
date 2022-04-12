import cv2
from os import listdir
from os import remove
from os.path import isfile, join

classs = "nb"

onlyfiles = [f for f in listdir("trainning_data/" + classs) if isfile(join("trainning_data/" + classs, f))]

for img in onlyfiles:
    print("image " + img)
    originalImage = cv2.imread('trainning_data/' + classs + "/" + img)
    dimensions = (224, 224)
    resized = cv2.resize(originalImage, dimensions, interpolation = cv2.INTER_AREA)
    cv2.imwrite('trainning_data/' + classs + "/" + img, resized)