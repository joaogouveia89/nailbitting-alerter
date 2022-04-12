import cv2
from os import listdir
from os import remove
from os.path import isfile, join
import numpy as np
import mediapipe as mp


classs = "nb"

onlyfiles = [f for f in listdir("trainning_data/" + classs) if isfile(join("trainning_data/" + classs, f))]


mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
idx = 0
for img in onlyfiles:
    idx = idx + 1
    if idx % 1000 == 0:
        print("index " + str(idx))
    originalImage = cv2.imread('trainning_data/' + classs + "/" + img)
    

    # flip the frame to horizontal direction
    image = cv2.flip(originalImage, 1)
    height , width, channel = image.shape

    RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # get the result
    results = selfie_segmentation.process(RGB)
    # extract segmented mask
    mask = results.segmentation_mask
    condition = np.stack(
    (results.segmentation_mask,) * 3, axis=-1) > 0.5

     # resize the background image to the same size of the original frame
    bg_image = blank_image = np.zeros((height,width,3), np.uint8)
    bg_image = cv2.resize(bg_image, (width, height))

    output_image = np.where(condition, image, bg_image)

    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite('trainning_data/' + classs + "/" + img, output_image)