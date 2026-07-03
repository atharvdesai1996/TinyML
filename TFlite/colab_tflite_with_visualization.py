"""
Cats vs. Dogs Image Classifier - Google Colab Interactive Version
This version includes automatic dependency management, visualization features,
and is optimized for Google Colab notebooks.

IMPORTANT: This code is designed for Google Colab. For production use,
see train_and_convert_to_tflite.py instead.
"""

import os
import subprocess
import sys

# Use Keras 2 (the API the course code was written against)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

needs_restart = False

# ==========================================
# COLAB-SPECIFIC SETUP
# ==========================================
# Colab's protobuf runtime is older than the gencode baked into its
# tensorflow_metadata, which breaks `import tensorflow_datasets`. Upgrade
# protobuf so runtime >= gencode. (One-time; triggers a session restart.)
import google.protobuf
_pb = tuple(int(p) for p in google.protobuf.__version__.split(".")[:2])
if _pb < (6, 31):
    print("Upgrading protobuf (fixes tensorflow_datasets import)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "--upgrade", "protobuf>=6.31.1"])
    needs_restart = True

# Ensure the remaining packages are installed
try:
    import tensorflow, tf_keras, tensorflow_hub, tensorflow_datasets  # noqa: F401
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "tensorflow",
                           "tf_keras",
                           "tensorflow_hub",
                           "tensorflow_datasets"])
    needs_restart = True

if needs_restart:
    print()
    print("=" * 60)
    print("Setup complete. Now: Runtime > Restart session, then re-run this cell.")
    print("=" * 60)
else:
    import tensorflow as tf
    import tf_keras
    import tensorflow_hub as hub
    import tensorflow_datasets as tfds
    print(f"TensorFlow {tf.__version__} (tf_keras {tf_keras.__version__})")
    print(f"tensorflow_hub {hub.__version__}, "
          f"tensorflow_datasets {getattr(tfds, '__version__', '?')}")

# ==========================================
# IMPORTS
# ==========================================
import numpy as np
import matplotlib.pylab as plt
import pathlib
from tqdm import tqdm

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# Fix for Kaggle dataset URL change
setattr(tfds.image_classification.cats_vs_dogs, '_URL',
        "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip")


# ==========================================
# DATASET PREPARATION
# ==========================================
def format_image(image, label):
    """Resizes images to 224x224 and normalizes pixel values."""
    image = tf.image.resize(image, (224, 224)) / 255.0
    return image, label


# Load the dataset and split it
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

num_examples = metadata.splits['train'].num_examples
num_classes = metadata.features['label'].num_classes
print(f"Number of examples: {num_examples}")
print(f"Number of classes: {num_classes}")

BATCH_SIZE = 32
train_batches = raw_train.shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = raw_validation.map(format_image).batch(BATCH_SIZE).prefetch(1)
test_batches = raw_test.map(format_image).batch(1)

# Verify batch shape
for image_batch, label_batch in train_batches.take(1):
    print(f"Image batch shape: {image_batch.shape}")

# ==========================================
# MODEL BUILDING (TRANSFER LEARNING)
# ==========================================
module_selection = ("mobilenet_v2", 224, 1280)
handle_base, pixels, FV_SIZE = module_selection
MODULE_HANDLE = "https://tfhub.dev/google/tf2-preview/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {} and output dimension {}".format(MODULE_HANDLE, IMAGE_SIZE, FV_SIZE))

feature_extractor = hub.KerasLayer(MODULE_HANDLE,
                                   input_shape=IMAGE_SIZE + (3,),
                                   output_shape=[FV_SIZE],
                                   trainable=False)

print("Building model with", MODULE_HANDLE)

model = tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ==========================================
# TRAINING
# ==========================================
EPOCHS = 5

hist = model.fit(train_batches,
                 epochs=EPOCHS,
                 validation_data=validation_batches)

# ==========================================
# SAVE AND EXPORT MODELS
# ==========================================
CATS_VS_DOGS_SAVED_MODEL = "exp_saved_model"
tf.saved_model.save(model, CATS_VS_DOGS_SAVED_MODEL)

converter = tf.lite.TFLiteConverter.from_saved_model(CATS_VS_DOGS_SAVED_MODEL)
tflite_model = converter.convert()

# Note: Using /tmp/ for Colab compatibility
tflite_models_dir = pathlib.Path("/tmp/")
tflite_model_file = tflite_models_dir / 'model1.tflite'
tflite_model_file.write_bytes(tflite_model)
print(f"TFLite model saved to: {tflite_model_file}")
print(f"File size: {tflite_model_file.stat().st_size / 1024 / 1024:.2f} MB")

# ==========================================
# TFLITE MODEL EVALUATION
# ==========================================
# Load TFLite model and allocate tensors
tflite_model_file = '/tmp/model1.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

predictions = []
test_labels, test_imgs = [], []

# Run inference on 100 test images
for img, label in tqdm(test_batches.take(100), desc="Running predictions"):
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    predictions.append(interpreter.get_tensor(output_index))
    
    test_labels.append(label.numpy()[0])
    test_imgs.append(img)

# Calculate accuracy
score = 0
for item in range(0, len(predictions)):
    prediction = np.argmax(predictions[item])
    label = test_labels[item]
    if prediction == label:
        score = score + 1

print(f"\nOut of 100 predictions I got {score} correct")

# ==========================================
# VISUALIZATION UTILITIES
# ==========================================
class_names = ['cat', 'dog']


def plot_image(i, predictions_array, true_label, img):
    """
    Plots a single image with its prediction and ground truth label.
    Color-codes the prediction: green if correct, red if incorrect.
    """
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    img = np.squeeze(img)
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


# ==========================================
# VISUALIZE PREDICTIONS
# ==========================================
# You can adjust max_index to view different predictions (1-100)
max_index = 10  # Change this value to see more or fewer predictions

for index in range(0, max_index):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(index, predictions, test_labels, test_imgs)
    plt.show()

print(f"\nDisplayed {max_index} predictions. Adjust 'max_index' variable to see more.")
