import cv2
import time
import argparse
import os
from datetime import datetime
import mediapipe as mp
import numpy as np


# Get the command line args
parser = argparse.ArgumentParser()
parser.add_argument("--classification", default='class1', help="nb for nail bitting or nnb for non nail bitting")

args = parser.parse_args()

dir = "trainning_data/" + args.classification

if args.classification not in ["nb", "nnb"] :
    print("Classification must be nb for nail bitting or nnb for non nail bitting")
    quit()

if not os.path.isdir(dir):
	os.makedirs(dir)

# Set Capture Device
cap = cv2.VideoCapture(0) 

path, dirs, files = next(os.walk(dir))

frame_num = len(files)

mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Warmup...
time.sleep(2)

start_time = datetime.now()
print("Capturing frames, started at " + start_time.strftime("%H:%M:%S"), "and will be done until reaches 10000 frames")
while(frame_num < 10000):
	# Capture frame-by-frame
	ret, frame = cap.read()
	cv2.imshow("getting images for " + args.classification + " - Press ESC to quit", frame)
	# flip the frame to horizontal direction
	image = cv2.flip(frame, 1)
	height , width, channel = image.shape

	RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# get the result
	results = selfie_segmentation.process(RGB)

	# extract segmented mask
	mask = results.segmentation_mask
	condition = np.stack(
		(results.segmentation_mask,) * 3, axis=-1) > 0.5

		# resize the background image to the same size of the original frame
	bg_image = np.zeros((height,width,3), np.uint8)
	bg_image = cv2.resize(bg_image, (width, height))

	output_image = np.where(condition, image, bg_image)

	cv2.imwrite(os.path.join(dir, str(frame_num) + ".jpg"), output_image)
	frame_num += 1
	time.sleep(0.6)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break


#closing all open windows 
cv2.destroyAllWindows() 

# close the already opened camera
cap.release()