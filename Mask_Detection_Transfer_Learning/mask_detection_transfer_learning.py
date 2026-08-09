"""
Mask Detection using Transfer Learning (MobileNet-V1)
=====================================================

This script trains a binary image classifier (with_mask vs without_mask) by
reusing a MobileNet-V1 backbone pretrained on ImageNet as a fixed feature
extractor and training only a small custom head on top.

Pipeline overview:
    Step 1  : Environment setup (Keras 2 compatibility, imports).
    Step 2  : Download and extract the dataset.
    Step 3  : Build tf.data pipelines for train / validation.
    Step 4  : Create a test split from the validation set.
    Step 5  : Prefetch for performance.
    Step 6  : Define data augmentation.
    Step 7  : Define input rescaling / preprocessing.
    Step 8  : Load MobileNet-V1 base (no top) and freeze it.
    Step 9  : Add a classification head.
    Step 10 : Assemble the full functional model.
    Step 11 : Compile the model.
    Step 12 : Evaluate the untrained model as a sanity check.
    Step 13 : Train the model.
    Step 14 : Plot training curves.
    Step 15 : Evaluate on the test set and visualize predictions.

Designed to run in Google Colab, but works locally with TensorFlow installed.
"""

# -----------------------------------------------------------------------------
# Step 1: Environment setup
# -----------------------------------------------------------------------------
# We force Keras 2 semantics because the course code (and the pretrained
# preprocessing helpers) were written against the Keras 2 API surface.
import os
import subprocess
import sys

os.environ["TF_USE_LEGACY_KERAS"] = "1"

try:
    import tensorflow as tf
    import tf_keras
    print(f"TensorFlow {tf.__version__} (tf_keras {tf_keras.__version__})")
except ImportError:
    # Install TF + tf_keras if we're on a fresh Colab runtime.
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "tensorflow", "tf_keras"])
    print("=" * 60)
    print("Install complete. Now: Runtime > Restart session, then re-run.")
    print("=" * 60)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory


# -----------------------------------------------------------------------------
# Step 2: Download and extract the dataset
# -----------------------------------------------------------------------------
# The dataset is a modified version of the Kaggle Face Mask Lite dataset,
# already split into train/ and validation/ folders with with_mask and
# without_mask subfolders. It lives on Google Drive; gdown handles the fetch.
#
# In a notebook, run the two shell commands below instead of subprocess:
#   !pip install --upgrade --no-cache-dir gdown
#   !gdown https://drive.google.com/uc?id=1lYOgCLLJU8TCIeTxJHkjsxBq_GPzQYb9
#   !unzip edx_transfer_learningv3.zip
#
# Uncomment the block below if you want the script to do this itself.
# subprocess.check_call([sys.executable, "-m", "pip", "install",
#                        "--upgrade", "--no-cache-dir", "gdown"])
# subprocess.check_call(["gdown",
#     "https://drive.google.com/uc?id=1lYOgCLLJU8TCIeTxJHkjsxBq_GPzQYb9"])
# subprocess.check_call(["unzip", "-q", "edx_transfer_learningv3.zip"])

# Point PATH at the extracted dataset root.
path_to_zip = "/content/"
PATH = os.path.join(os.path.dirname(path_to_zip),
                    'edx_transfer_learningv3/edx_transfer_learning/')
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')


# -----------------------------------------------------------------------------
# Step 3: Build tf.data pipelines
# -----------------------------------------------------------------------------
# image_dataset_from_directory auto-labels images based on subfolder names
# (with_mask -> 0, without_mask -> 1, alphabetically). We resize to 96x96 to
# keep the model TinyML-friendly.
BATCH_SIZE = 32
IMG_SIZE = (96, 96)

train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)

class_names = train_dataset.class_names
print("Classes:", class_names)

# Quick visual sanity check — display 9 training images with their labels.
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


# -----------------------------------------------------------------------------
# Step 4: Create a test split from the validation set
# -----------------------------------------------------------------------------
# The dataset ships with train/ and validation/ only. We carve out ~20% of the
# validation batches as a proper held-out test set the model never sees during
# training or validation.
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

print('Number of validation batches: %d'
      % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d'
      % tf.data.experimental.cardinality(test_dataset))


# -----------------------------------------------------------------------------
# Step 5: Prefetch for performance
# -----------------------------------------------------------------------------
# Prefetching overlaps data loading with model execution on the accelerator so
# the GPU/TPU never sits idle waiting for the next batch.
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


# -----------------------------------------------------------------------------
# Step 6: Data augmentation
# -----------------------------------------------------------------------------
# Small dataset -> we need to synthetically expand it. Random horizontal flips
# and small rotations produce realistic new views of each face. These layers
# are ACTIVE only during model.fit(); they no-op during evaluate/predict.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# Preview augmentation on a single image.
for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')


# -----------------------------------------------------------------------------
# Step 7: Input rescaling / preprocessing
# -----------------------------------------------------------------------------
# MobileNet expects pixel values in [-1, 1] but our tensors are still [0, 255].
# tf.keras.applications.mobilenet.preprocess_input performs exactly that
# normalization. (The manual Rescaling layer below is an equivalent alternative
# and is kept here as a reference.)
preprocess_input = tf.keras.applications.mobilenet.preprocess_input
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5,
                                                               offset=-1)


# -----------------------------------------------------------------------------
# Step 8: Load the MobileNet-V1 base and freeze it
# -----------------------------------------------------------------------------
# include_top=False removes the original ImageNet 1000-class classifier so we
# can attach our own head. weights='imagenet' loads the pretrained parameters.
# We then set trainable=False so gradients don't flow into the backbone during
# training — this is the "feature extraction" strategy of transfer learning.
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE,
                                             include_top=False,
                                             weights='imagenet')

# Sanity check: what feature map does the backbone produce?
image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print("Backbone output shape:", feature_batch.shape)

base_model.trainable = False
base_model.summary()


# -----------------------------------------------------------------------------
# Step 9: Classification head
# -----------------------------------------------------------------------------
# GlobalAveragePooling2D collapses the HxWxC feature map into a single
# C-dimensional vector per image. A single Dense(1) unit with a linear
# activation produces the logit for our binary classifier.
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print("Pooled feature shape:", feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print("Prediction shape:", prediction_batch.shape)


# -----------------------------------------------------------------------------
# Step 10: Assemble the full model (Functional API)
# -----------------------------------------------------------------------------
# Data flow: input -> augment -> preprocess -> frozen MobileNet -> pool ->
#            dropout -> dense logit
# training=False on base_model keeps its BatchNorm layers in inference mode,
# which is important when the backbone is frozen.
inputs = tf.keras.Input(shape=(96, 96, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)  # regularization
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)


# -----------------------------------------------------------------------------
# Step 11: Compile
# -----------------------------------------------------------------------------
# BinaryCrossentropy(from_logits=True) is the numerically stable pairing with
# a linear output. Adam with a modest learning rate is a safe default.
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()


# -----------------------------------------------------------------------------
# Step 12: Baseline evaluation (untrained head)
# -----------------------------------------------------------------------------
# Before any training we expect accuracy ~50% (random) because the Dense head
# has random weights, even though the backbone features are meaningful.
loss0, accuracy0 = model.evaluate(validation_dataset)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))


# -----------------------------------------------------------------------------
# Step 13: Train
# -----------------------------------------------------------------------------
# Because the backbone is frozen, only the tiny head is being learned. This
# converges extremely quickly — 10 epochs is plenty, and in favorable
# initializations 2-5 epochs can already exceed 95% val accuracy.
EPOCHS = 10
history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    validation_data=validation_dataset)


# -----------------------------------------------------------------------------
# Step 14: Plot training curves
# -----------------------------------------------------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# -----------------------------------------------------------------------------
# Step 15: Test-set evaluation + qualitative predictions
# -----------------------------------------------------------------------------
loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)

# Pull one batch and run predictions.
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# The model outputs logits; apply sigmoid then threshold at 0.5.
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

# Visualize predicted class names over the images.
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[predictions[i]])
    plt.axis("off")

# -----------------------------------------------------------------------------
# Notes on epoch selection
# -----------------------------------------------------------------------------
# With a fully frozen MobileNet-V1 backbone and only ~1K trainable parameters
# in the head, this task converges almost immediately. EPOCHS = 10 gives a
# reliable, reproducible result; going higher yields diminishing returns and
# can start to overfit the small dataset. This is the payoff of transfer
# learning: strong accuracy, tiny training budget.
