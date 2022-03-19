import numpy as np
import os

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
    __img_height = 180
    __img_width = 180
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
            for image_batch, labels_batch in self.__dataset_tranning:
                print(image_batch.shape)
                print(labels_batch.shape)
                break
            
            self.__dataset_tranning = self.__dataset_tranning.cache().shuffle(1000).prefetch(buffer_size=self.__AUTOTUNE)
            self. __dataset_validation = self. __dataset_validation.cache().prefetch(buffer_size=self.__AUTOTUNE)

            normalization_layer = layers.Rescaling(1./255)

            normalized_ds = self.__dataset_tranning.map(lambda x, y: (normalization_layer(x), y))
            image_batch, labels_batch = next(iter(normalized_ds))
            first_image = image_batch[0]
            # Notice the pixel values are now in `[0,1]`.
            print(np.min(first_image), np.max(first_image))

            model = keras.models.Sequential([
                layers.Rescaling(1./255, input_shape=(self.__img_height, self.__img_width, 3)),
                layers.Conv2D(16, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(32, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(2)
            ])

            model.compile(optimizer='adam',
                            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])

            model.summary()

            history = model.fit(
                self.__dataset_tranning,
                validation_data=self. __dataset_validation,
                epochs = self.__training_epochs
            )
            model.save(self.__model_file)

    # this method returns the label and the confidence in percentage
    def get_prediction(self, frame):

        # this is to have the model already loaded in memory whenever this method is required more than once
        if(self.__model_instance == None):
            self.__model_instance = self.__get_model()

        # Resizing into dimensions used while training
        input = frame.resize((self.__img_height, self.__img_width))
        input_array = np.array(input)
        input_array = expand_dims(input_array, 0) # Create a batch

        pred = self.__model_instance.predict(input_array)
        score = nn.softmax(pred[0])

        return self.__class_names[np.argmax(score)], 100 * np.max(score)