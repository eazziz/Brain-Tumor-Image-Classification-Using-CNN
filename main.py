#Importing Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score
import os
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import Precision, Recall
from keras.layers import Dense, Input, Flatten, Reshape
from keras.models import Model

#DATA PREPROCESSING

#This function essentially adds some variability or randomness to our images as well as scales the pixel values from 0-255 to 0-1. The function as a whole is meant to prevent overfitting.
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

#Here, we are applies the function from above onto our dataset. We also resize the images into 64x64 as well as define our batch size for each epoch to 32.
training_set = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/data/brain-tumor-mri-dataset/Training',
    target_size=(64, 64),
    batch_size=64,
    class_mode='categorical'
)

#This function only scales the pixel values as opposed to adding variability, as seen previously.
test_datagen = ImageDataGenerator(
    rescale = 1./255
)

test_set = test_datagen.flow_from_directory(
    '/content/drive/MyDrive/data/brain-tumor-mri-dataset/Testing',
    target_size = (64, 64),
    batch_size=64,
    class_mode = 'categorical'
)

#This function is meant to drop any dupliucates. Additionally, the ImageDataGenerator automatically removes any corrupted images.
def remove_duplicates(directory):
    seen = set()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            if filename in seen:
                print(f"Duplicate found: {filename}. Deleting.")
                os.remove(file_path)  # Remove duplicate
            else:
                seen.add(filename)

remove_duplicates('/content/drive/MyDrive/data/brain-tumor-mri-dataset/Training/glioma')
remove_duplicates('/content/drive/MyDrive/data/brain-tumor-mri-dataset/Training/meningioma')
remove_duplicates('/content/drive/MyDrive/data/brain-tumor-mri-dataset/Training/notumor')
remove_duplicates('/content/drive/MyDrive/data/brain-tumor-mri-dataset/Training/pituitary')

remove_duplicates('/content/drive/MyDrive/data/brain-tumor-mri-dataset/Testing/glioma')
remove_duplicates('/content/drive/MyDrive/data/brain-tumor-mri-dataset/Testing/glioma')
remove_duplicates('/content/drive/MyDrive/data/brain-tumor-mri-dataset/Testing/glioma')
remove_duplicates('/content/drive/MyDrive/data/brain-tumor-mri-dataset/Testing/glioma')

# Path to the training folder
training_path = '/content/drive/MyDrive/data/brain-tumor-mri-dataset/Training'

# Get the list of subfolders (class names)
class_names = sorted(os.listdir(training_path))

# Map class names to numerical labels
class_mapping = {class_name: idx for idx, class_name in enumerate(class_names)}

# Display the mapping
print("Class Mapping:")
for class_name, label in class_mapping.items():
    print(f"'{class_name}' is encoded as {label}")

import matplotlib.pyplot as plt
import os

# Path to the training folder
training_path = '/content/drive/MyDrive/data/brain-tumor-mri-dataset/Training'

# Get the list of subfolders (class names)
class_names = sorted(os.listdir(training_path))

# Get the number of images for each class
class_counts = [len(os.listdir(os.path.join(training_path, class_name))) for class_name in class_names]

# Create the class distribution plot
plt.figure(figsize=(10, 6))
plt.bar(class_names, class_counts, color='skyblue')
plt.xlabel('Class Names')
plt.ylabel('Number of Images')
plt.title('Class Distribution in the Training Dataset')
plt.show()

encoder = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=[64, 64, 3]),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
])

#Initliaze our model
cnn = tf.keras.Sequential([
    encoder,  # Reuse the encoder from the autoencoder
    tf.keras.layers.Flatten(),  # Flatten the encoded representation
    tf.keras.layers.Dense(units=128, activation='relu'),  # Fully connected layer
    tf.keras.layers.Dense(units=4, activation='softmax')  # Classification layer
])

# Compile the models
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy', Precision(), Recall(), 'AUC'])

# Summary of models
cnn.summary()


#Model hyperparameters
N_TRAIN_SAMPLES = 5800
N_TEST_SAMPLES = 1300

train_batch_size = 32
test_batch_size = 32

train_steps = N_TRAIN_SAMPLES // train_batch_size
test_steps = N_TEST_SAMPLES // test_batch_size

# Train the classifier
cnn.fit(x=training_set, validation_data=test_set, steps_per_epoch=train_steps, validation_steps=test_steps, epochs=10)

#Evaluate the model on the test set
# Get true labels and predictions
test_labels = []
test_predictions = []
for i in range(len(test_set)):
    batch_images, batch_labels = test_set[i]
    test_labels.extend(batch_labels)
    batch_predictions = cnn.predict(batch_images)
    test_predictions.extend(np.argmax(batch_predictions, axis=1))

test_labels = np.array(test_labels)
test_predictions = np.array(test_predictions)

# Compute metrics
test_loss, test_accuracy, test_precision, test_recall, test_auc = cnn.evaluate(test_set, verbose=0)

# Print all metrics
print("Final Metrics:")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")
print(f"Test AUC: {test_auc}")
