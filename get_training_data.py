import cv2
import time
import argparse
import os
from datetime import datetime

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

# Warmup...
time.sleep(2)

start_time = datetime.now()
print("Capturing frames, started at " + start_time.strftime("%H:%M:%S"), "and will be done until reaches 10000 frames")
while(frame_num < 100):
	# Capture frame-by-frame
	ret, frame = cap.read()
	cv2.imshow("getting images for " + args.classification + " - Press ESC to quit", frame)
	frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)

	cv2.imwrite(os.path.join(dir, str(frame_num) + ".jpg"), frame)
	frame_num += 1
	time.sleep(0.6)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break


#closing all open windows 
cv2.destroyAllWindows() 

# close the already opened camera
cap.release()