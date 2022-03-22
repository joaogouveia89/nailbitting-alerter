import cv2
import time
from PIL import Image

class NailbittingMonitor:
    keep_running = True

    def __init__(self, interval):
        self.__interval = interval
    
    def start(self, callback):
        vid = cv2.VideoCapture(0)
        while(self.keep_running):
            _, frame = vid.read()
            callback(frame)
            time.sleep(self.__interval)
        vid.release()