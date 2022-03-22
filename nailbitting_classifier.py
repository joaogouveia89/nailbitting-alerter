import numpy as np
import os
import cv2

import tensorflow_hub as tfh
import tensorflow as tf
from tensorflow import keras
from tensorflow import data
from tensorflow import expand_dims
from tensorflow import nn
from keras import layers
from nailbitting_classification import NailbittingClassification

import pathlib

# This class labels the input image in nailbitting or nonnailbitting according to the trainning data provided by the user in __data_path
class NailbittingClassifier:
    __model_file = "nail_bitting_model"
    __data_path = "trainning_data"

    __data_pathlib = pathlib.Path(__data_path)
    __batch_size = 32
    __img_height = 320
    __img_width = 140
    __training_epochs = 10
    __AUTOTUNE = data.AUTOTUNE

    __model_instance = None

    __dataset_tranning = keras.utils.image_dataset_from_directory(
                __data_pathlib,
                validation_split=0.2,
                subset="training",
                seed=123,
                image_size=(__img_height, __img_width),
                batch_size=__batch_size)
    
    __dataset_validation = keras.utils.image_dataset_from_directory(
                __data_pathlib,
                validation_split=0.2,
                subset="validation",
                seed=123,
                image_size=(__img_height, __img_width),
                batch_size=__batch_size)

    __class_names = [NailbittingClassification.BITING, NailbittingClassification.NON_BITING]
    
    def __get_model(self):
        if os.path.exists(self.__model_file) == True:
            return  keras.models.load_model(self.__model_file)
        else:
            IMAGE_SIZE = (384, 384)

            model = keras.models.Sequential([
                keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
                tfh.KerasLayer("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/feature_vector/2", trainable=True),
                keras.layers.Dropout(rate=0.2),
                keras.layers.Dense(len(self.__class_names),
                                    kernel_regularizer=tf.keras.regularizers.l2(0.0001))
            ])

            model.build((None,)+IMAGE_SIZE+(3,))

            model.summary()

            model.compile(
                optimizer=keras.optimizers.SGD(learning_rate=0.005, momentum=0.9), 
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

            steps_per_epoch = len(self.__dataset_tranning) 
            validation_steps = len(self.__dataset_validation) 

            model.fit(
                 self.__dataset_tranning,
                epochs=5, steps_per_epoch=steps_per_epoch,
                validation_data=self.__dataset_validation,
                validation_steps=validation_steps)
            model.save(self.__model_file)

            return model

    # this method returns the label and the confidence in percentage
    def get_prediction(self, frame):

        # this is to have the model already loaded in memory whenever this method is required more than once
        if(self.__model_instance == None):
            self.__model_instance = self.__get_model()

        # Resizing into dimensions used while training
        input_array = cv2.resize(frame, (self.__img_height, self.__img_width), interpolation = cv2.INTER_AREA)

        input_array = expand_dims(input_array, 0) # Create a batch

        print("STARTING PREDICT")

        pred = self.__model_instance.predict(input_array)

        score = nn.softmax(pred[0])

        print("score = " + str(score))

        return self.__class_names[np.argmax(score)], 100 * np.max(score)