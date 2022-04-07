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

model_handle_map = {
  "efficientnetv2-s": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/feature_vector/2",
  "efficientnetv2-m": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_m/feature_vector/2",
  "efficientnetv2-l": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/feature_vector/2",
  "efficientnetv2-s-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/feature_vector/2",
  "efficientnetv2-m-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_m/feature_vector/2",
  "efficientnetv2-l-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_l/feature_vector/2",
  "efficientnetv2-xl-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/feature_vector/2",
  "efficientnetv2-b0-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2",
  "efficientnetv2-b1-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b1/feature_vector/2",
  "efficientnetv2-b2-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b2/feature_vector/2",
  "efficientnetv2-b3-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b3/feature_vector/2",
  "efficientnetv2-s-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_s/feature_vector/2",
  "efficientnetv2-m-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_m/feature_vector/2",
  "efficientnetv2-l-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_l/feature_vector/2",
  "efficientnetv2-xl-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2",
  "efficientnetv2-b0-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b0/feature_vector/2",
  "efficientnetv2-b1-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b1/feature_vector/2",
  "efficientnetv2-b2-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b2/feature_vector/2",
  "efficientnetv2-b3-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/feature_vector/2",
  "efficientnetv2-b0": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2",
  "efficientnetv2-b1": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b1/feature_vector/2",
  "efficientnetv2-b2": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b2/feature_vector/2",
  "efficientnetv2-b3": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b3/feature_vector/2",
  "efficientnet_b0": "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
  "efficientnet_b1": "https://tfhub.dev/tensorflow/efficientnet/b1/feature-vector/1",
  "efficientnet_b2": "https://tfhub.dev/tensorflow/efficientnet/b2/feature-vector/1",
  "efficientnet_b3": "https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1",
  "efficientnet_b4": "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1",
  "efficientnet_b5": "https://tfhub.dev/tensorflow/efficientnet/b5/feature-vector/1",
  "efficientnet_b6": "https://tfhub.dev/tensorflow/efficientnet/b6/feature-vector/1",
  "efficientnet_b7": "https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1",
  "bit_s-r50x1": "https://tfhub.dev/google/bit/s-r50x1/1",
  "inception_v3": "https://tfhub.dev/google/imagenet/inception_v3/feature-vector/4",
  "inception_resnet_v2": "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature-vector/4",
  "resnet_v1_50": "https://tfhub.dev/google/imagenet/resnet_v1_50/feature-vector/4",
  "resnet_v1_101": "https://tfhub.dev/google/imagenet/resnet_v1_101/feature-vector/4",
  "resnet_v1_152": "https://tfhub.dev/google/imagenet/resnet_v1_152/feature-vector/4",
  "resnet_v2_50": "https://tfhub.dev/google/imagenet/resnet_v2_50/feature-vector/4",
  "resnet_v2_101": "https://tfhub.dev/google/imagenet/resnet_v2_101/feature-vector/4",
  "resnet_v2_152": "https://tfhub.dev/google/imagenet/resnet_v2_152/feature-vector/4",
  "nasnet_large": "https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/4",
  "nasnet_mobile": "https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4",
  "pnasnet_large": "https://tfhub.dev/google/imagenet/pnasnet_large/feature_vector/4",
  "mobilenet_v2_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
  "mobilenet_v2_130_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/4",
  "mobilenet_v2_140_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4",
  "mobilenet_v3_small_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5",
  "mobilenet_v3_small_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/feature_vector/5",
  "mobilenet_v3_large_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5",
  "mobilenet_v3_large_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/feature_vector/5",
}

model_image_size_map = {
  "efficientnetv2-s": 384,
  "efficientnetv2-m": 480,
  "efficientnetv2-l": 480,
  "efficientnetv2-b0": 224,
  "efficientnetv2-b1": 240,
  "efficientnetv2-b2": 260,
  "efficientnetv2-b3": 300,
  "efficientnetv2-s-21k": 384,
  "efficientnetv2-m-21k": 480,
  "efficientnetv2-l-21k": 480,
  "efficientnetv2-xl-21k": 512,
  "efficientnetv2-b0-21k": 224,
  "efficientnetv2-b1-21k": 240,
  "efficientnetv2-b2-21k": 260,
  "efficientnetv2-b3-21k": 300,
  "efficientnetv2-s-21k-ft1k": 384,
  "efficientnetv2-m-21k-ft1k": 480,
  "efficientnetv2-l-21k-ft1k": 480,
  "efficientnetv2-xl-21k-ft1k": 512,
  "efficientnetv2-b0-21k-ft1k": 224,
  "efficientnetv2-b1-21k-ft1k": 240,
  "efficientnetv2-b2-21k-ft1k": 260,
  "efficientnetv2-b3-21k-ft1k": 300, 
  "efficientnet_b0": 224,
  "efficientnet_b1": 240,
  "efficientnet_b2": 260,
  "efficientnet_b3": 300,
  "efficientnet_b4": 380,
  "efficientnet_b5": 456,
  "efficientnet_b6": 528,
  "efficientnet_b7": 600,
  "inception_v3": 299,
  "inception_resnet_v2": 299,
  "nasnet_large": 331,
  "pnasnet_large": 331,
}

# This class labels the input image in nailbitting or nonnailbitting according to the trainning data provided by the user in __data_path
class NailbittingClassifier:
    __model_file = "nail_bitting_model"
    __data_path = "trainning_data"

    __data_pathlib = pathlib.Path(__data_path)
    __batch_size = 32
    __img_height = 224
    __img_width = 224
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

    def __build_dataset(self, subset, im_size):
      return tf.keras.preprocessing.image_dataset_from_directory(
          self.__data_path,
          validation_split=.20,
          subset=subset,
          label_mode="categorical",
          # Seed needs to provided when using validation_split and shuffle = True.
          # A fixed seed is used so that the validation set is stable across runs.
          seed=123,
          image_size=im_size,
          batch_size=1)
    
    def __get_model(self):
        if os.path.exists(self.__model_file) == True:
            return  keras.models.load_model(self.__model_file)
        else:
            model_name = "mobilenet_v2_100_224"
            model_handle = model_handle_map.get(model_name)
            pixels = model_image_size_map.get(model_name, 224)

            print(f"Selected model: {model_name} : {model_handle}")

            IMAGE_SIZE = (pixels, pixels)
            print(f"Input size {IMAGE_SIZE}")

            BATCH_SIZE = 16

            do_fine_tuning = False

            train_ds = self.__build_dataset("training", IMAGE_SIZE)

            train_size = train_ds.cardinality().numpy()
            train_ds = train_ds.unbatch().batch(BATCH_SIZE)
            train_ds = train_ds.repeat()

            normalization_layer = tf.keras.layers.Rescaling(1. / 255)
            preprocessing_model = tf.keras.Sequential([normalization_layer])
            do_data_augmentation = False #@param {type:"boolean"}
            if do_data_augmentation:
              preprocessing_model.add(
                  tf.keras.layers.RandomRotation(40))
              preprocessing_model.add(
                  tf.keras.layers.RandomTranslation(0, 0.2))
              preprocessing_model.add(
                  tf.keras.layers.RandomTranslation(0.2, 0))
              # Like the old tf.keras.preprocessing.image.ImageDataGenerator(),
              # image sizes are fixed when reading, and then a random zoom is applied.
              # If all training inputs are larger than image_size, one could also use
              # RandomCrop with a batch size of 1 and rebatch later.
              preprocessing_model.add(
                  tf.keras.layers.RandomZoom(0.2, 0.2))
              preprocessing_model.add(
                  tf.keras.layers.RandomFlip(mode="horizontal"))
            train_ds = train_ds.map(lambda images, labels:
                                    (preprocessing_model(images), labels))

            val_ds = self.__build_dataset("validation", IMAGE_SIZE)
            valid_size = val_ds.cardinality().numpy()
            val_ds = val_ds.unbatch().batch(BATCH_SIZE)
            val_ds = val_ds.map(lambda images, labels:
                                (normalization_layer(images), labels))

            print("Building model with", model_handle)
            model = tf.keras.Sequential([
                # Explicitly define the input shape so the model can be properly
                # loaded by the TFLiteConverter
                tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
                hub.KerasLayer(model_handle, trainable=do_fine_tuning),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(len(self.__class_names),
                                      kernel_regularizer=tf.keras.regularizers.l2(0.0001))
            ])
            model.build((None,)+IMAGE_SIZE+(3,))
            model.summary()

            model.compile(
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9), 
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
              metrics=['accuracy'])

            steps_per_epoch = train_size // BATCH_SIZE
            validation_steps = valid_size // BATCH_SIZE
            model.fit(
                train_ds,
                epochs=5, steps_per_epoch=steps_per_epoch,
                validation_data=val_ds,
                validation_steps=validation_steps)

            model.save(self.__model_file)

            return model

    # this method returns the label and the confidence in percentage
    def get_prediction(self, frame):

        # this is to have the model already loaded in memory whenever this method is required more than once
        if(self.__model_instance == None):
            self.__model_instance = self.__get_model()
        
        dimensions = frame.shape

        editted = frame[0: dimensions[0], 0:480]
        # Resizing into dimensions used while training
        input_array = cv2.resize(editted, (self.__img_height, self.__img_width), interpolation = cv2.INTER_AREA)

        input_array = expand_dims(input_array, 0) # Create a batch

        print("STARTING PREDICT")

        pred = self.__model_instance.predict(input_array)

        score = nn.softmax(pred[0])

        print("score = " + str(score))

        return self.__class_names[np.argmax(score)], 100 * np.max(score)


# redimen all images on training_data to 224x224 and handle input to this dimen too