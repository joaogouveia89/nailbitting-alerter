import numpy as np
import os
import cv2

import tensorflow_hub as hub
import tensorflow as tf
from tensorflow import keras
from tensorflow import data
from tensorflow import expand_dims
from tensorflow import nn
from keras import layers
from nailbitting_classification import NailbittingClassification

import pathlib

# https://github.com/tensorflow/hub/tree/master/tensorflow_hub/tools/make_image_classifier to train the model
# This class labels the input image in nailbitting or nonnailbitting according to the trainning data provided by the user in __data_path
class NailbittingClassifier:
    __model_file = "nailbitting_model"

    __img_height = 224
    __img_width = 224

    __model_instance = None


    __class_names = [NailbittingClassification.BITING, NailbittingClassification.NON_BITING]
    
    def __get_model(self):
        if os.path.exists(self.__model_file) == True:
            return  tf.saved_model.load(self.__model_file)
        else:
            raise Exception("You must run make_image_classifier before")

    # this method returns the label and the confidence in percentage
    def get_prediction(self, frame):

        # this is to have the model already loaded in memory whenever this method is required more than once
        if(self.__model_instance == None):
            self.__model_instance = self.__get_model()

        # Resizing into dimensions used while training
        input_array = cv2.resize(frame, (self.__img_height, self.__img_width), interpolation = cv2.INTER_AREA)

        cv2.imwrite("img.jpg", input_array)

        input_array = input_array.astype('float32')

        input_array = expand_dims(input_array, 0) # Create a batch

        infer = self.__model_instance.signatures["serving_default"]
        labeling = infer(tf.constant(input_array))['dense']

        return self.__class_names[np.argmax(labeling)], 100 * np.max(labeling)


# redimen all images on training_data to 224x224 and handle input to this dimen too

#https://www.tensorflow.org/guide/saved_model