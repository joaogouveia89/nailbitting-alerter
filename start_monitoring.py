# https://www.tensorflow.org/tutorials/keras/save_and_load next


from nailbitting_classifier import NailbittingClassifier
from nailbitting_monitor import NailbittingMonitor
from nailbitting_classification import NailbittingClassification
import os
import cv2
#from win10toast import ToastNotifier
from datetime import datetime

def generate_alert(frame):
  if not os.path.exists('report'):
    os.makedirs('report')
  cv2.imwrite("report/nb.jpg", frame)
  
  now = datetime.now()
  html_raw = '<html><body><h1>Peguei voce roendo unha - ' + now.strftime("%d/%m/%Y %H:%M:%S") + '</h1><br><img src="nb.jpg" /></body></html>'
  with open('report/report.html', 'w') as file:
    file.write(html_raw)
  #toaster.show_toast("AVISO","PARE DE ROER UNHA")

def check_frame(frame):
  label, confidence = classifier.get_prediction(frame)
  to_prt = "bitting" if label == NailbittingClassification.BITING  else "non bitting"
  print("label is " + to_prt + " with accuracy " + str(confidence) + "%")
  if label == NailbittingClassification.BITING and confidence >= 90:
    generate_alert(frame)

monitor = NailbittingMonitor(3)
classifier = NailbittingClassifier()
#toaster = ToastNotifier()

monitor.start(check_frame)


# to study https://www.tensorflow.org/tutorials/keras/save_and_load