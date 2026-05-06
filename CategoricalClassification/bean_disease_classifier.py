import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

# Download and extract datasets (uncomment if running in Colab)
# !gdown "https://storage.googleapis.com/learning-datasets/beans/train.zip" -O /tmp/train.zip
# !gdown "https://storage.googleapis.com/learning-datasets/beans/validation.zip" -O /tmp/validation.zip
# !gdown "https://storage.googleapis.com/learning-datasets/beans/test.zip" -O /tmp/test.zip

# Extract datasets
for name in ['train', 'validation', 'test']:
    local_zip = f'/tmp/{name}.zip'
    if os.path.exists(local_zip):
        zip_ref = zipfile.ZipFile(local_zip, 'r')
        zip_ref.extractall('/tmp/')
        zip_ref.close()

# Data generators with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

TRAIN_DIRECTORY_LOCATION = '/tmp/train'
VAL_DIRECTORY_LOCATION = '/tmp/validation'
TARGET_SIZE = (224, 224)
CLASS_MODE = 'categorical'

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIRECTORY_LOCATION,
    target_size=TARGET_SIZE,
    batch_size=128,
    class_mode=CLASS_MODE
)

validation_generator = validation_datagen.flow_from_directory(
    VAL_DIRECTORY_LOCATION,
    target_size=TARGET_SIZE,
    batch_size=128,
    class_mode=CLASS_MODE
)

# Model definition
model = tf.keras.models.Sequential([
    # Find the features with Convolutions and Pooling
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Output layer: 3 classes (healthy, bean rust, angular leaf spots)
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()

OPTIMIZER = 'adam'
LOSS_FUNCTION = 'categorical_crossentropy'

model.compile(
    loss=LOSS_FUNCTION,
    optimizer=OPTIMIZER,
    metrics=['accuracy']
)

NUM_EPOCHS = 20

history = model.fit(
    train_generator,
    epochs=NUM_EPOCHS,
    verbose=1,
    validation_data=validation_generator)

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.xlim([0, NUM_EPOCHS])
plt.ylim([0.4, 1.0])
plt.show()
