import cv2
from os import listdir
from os import remove
from os.path import isfile, join

classs = "nb"

onlyfiles = [f for f in listdir("trainning_data/" + classs) if isfile(join("trainning_data/" + classs, f))]

for img in onlyfiles:
    print("image " + img)
    originalImage = cv2.imread('trainning_data/' + classs + "/" + img)
    dimensions = originalImage.shape
    remove('trainning_data/' + classs + "/" + img)
    editted = originalImage[0: dimensions[0], 0:480]
    cv2.imwrite('trainning_data/' + classs + "/" + img, editted)