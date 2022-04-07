import tensorflow as tf
import cv2

model = tf.keras.models.load_model("nb_model")

frame = cv2.imread("test.jpg")

class_names = ["nail bitting", "non nail bitting"]

print(model.signatures.keys())
