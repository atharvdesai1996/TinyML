"""
Cats vs. Dogs Image Classifier using Transfer Learning and TensorFlow Lite.
This script downloads the dataset, trains a MobileNetV2 feature extractor,
saves the model, and converts it to a highly optimized .tflite format.
"""

import os
# Fix to ensure compatibility between TensorFlow Hub and newer TensorFlow versions
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tqdm import tqdm

# ==========================================
# 1. DATASET PREPARATION
# ==========================================
print("Downloading and formatting the dataset...")
# Temporary fix for Kaggle dataset URL change
setattr(tfds.image_classification.cats_vs_dogs, '_URL', "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip")

def format_image(image, label):
    """Resizes images to 224x224 and normalizes pixel values."""
    image = tf.image.resize(image, (224, 224)) / 255.0
    return image, label

# Load the dataset and split it into training, validation, and testing sets
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

num_examples = metadata.splits['train'].num_examples
num_classes = metadata.features['label'].num_classes
BATCH_SIZE = 32

# Prepare batches for training and evaluation
train_batches = raw_train.shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = raw_validation.map(format_image).batch(BATCH_SIZE).prefetch(1)
test_batches = raw_test.map(format_image).batch(1)

# ==========================================
# 2. MODEL BUILDING (TRANSFER LEARNING)
# ==========================================
print("\nBuilding the model with MobileNetV2...")
MODULE_HANDLE = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
IMAGE_SIZE = (224, 224)
FV_SIZE = 1280

# Load the pre-trained feature extractor from TensorFlow Hub
feature_extractor = hub.KerasLayer(MODULE_HANDLE,
                                   input_shape=IMAGE_SIZE + (3,),
                                   output_shape=[FV_SIZE],
                                   trainable=False) # Freeze the base layer weights

# Attach a new classification head
model = tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ==========================================
# 3. TRAINING
# ==========================================
print("\nStarting training...")
EPOCHS = 5
hist = model.fit(train_batches,
                 epochs=EPOCHS,
                 validation_data=validation_batches)

# ==========================================
# 4. SAVE AND EXPORT MODELS
# ==========================================
print("\nSaving the standard TensorFlow model...")
CATS_VS_DOGS_SAVED_MODEL = "exp_saved_model"
tf.saved_model.save(model, CATS_VS_DOGS_SAVED_MODEL)

print("Converting the model to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_saved_model(CATS_VS_DOGS_SAVED_MODEL)
tflite_model = converter.convert()

# Create a directory for the TFLite model and save it
tflite_models_dir = pathlib.Path("./tflite_models")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir / 'model1.tflite'
tflite_model_file.write_bytes(tflite_model)
print(f"TFLite model successfully saved to: {tflite_model_file}")

# ==========================================
# 5. TFLITE MODEL EVALUATION
# ==========================================
print("\nEvaluating the TFLite model using the Interpreter...")
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

predictions = []
test_labels, test_imgs = [], []

# Run inference on 100 test images
for img, label in tqdm(test_batches.take(100), desc="Running Predictions"):
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    predictions.append(interpreter.get_tensor(output_index))
    test_labels.append(label.numpy()[0])
    test_imgs.append(img)

# Calculate Accuracy
score = 0
for item in range(0, len(predictions)):
    prediction = np.argmax(predictions[item])
    label = test_labels[item]
    if prediction == label:
        score += 1

print(f"\nResults: Out of 100 test predictions, the TFLite model got {score} correct!")
