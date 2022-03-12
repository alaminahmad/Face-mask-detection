# Facial expression recognition 
# Face mask detection software with Opencv, Tensorflow and Keras
import tensorflow as tf
import numpy as np
tf.__version__

width = 224
height = 224
#height and width are (224,224) since we will use the mobileNetv2 network
batch_size = 32
data_dir = r'link to your dataset'
# /content/dataset

training = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset='training',
    seed=123,
    image_size=(height, width),
    batch_size=batch_size
)

validation = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset='validation',
    seed=123,
    image_size=(height, width),
    batch_size=batch_size
)

classes = training.class_names

import matplotlib.pyplot as plt
for images, labels in training.take(1):
    plt.imshow(images[1].numpy().astype('uint8'))
    plt.title(classes[labels[1]])
    
from tensorflow.keras.applications import MobileNetV2
model = MobileNetV2(weights='imagenet')

model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])
model.summary()

face_mask_detection = model.fit(training,validation_data=validation,epochs=3)

