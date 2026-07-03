"""
TensorFlow Lite Quantization Assignment
========================================
This script demonstrates three different model optimization approaches:
1. Model 1: No optimization (baseline)
2. Model 2: Dynamic range quantization
3. Model 3: Full integer quantization with representative dataset

Quantization reduces model size and improves inference speed by converting
32-bit floating point weights to 8-bit integers.
"""

import os
import subprocess
import sys

# Use Keras 2 (the API the course code was written against)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

needs_restart = False

# ==========================================
# COLAB ENVIRONMENT SETUP
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

# ==========================================
# DATA PREPARATION
# ==========================================
def format_image(image, label):
    """
    Preprocesses images for the MobileNetV2 model.
    - Resizes to 224x224 (MobileNetV2 input size)
    - Normalizes pixel values from [0, 255] to [0, 1]
    """
    image = tf.image.resize(image, (224, 224)) / 255.0
    return image, label

# Load the Cats vs Dogs dataset from TensorFlow Datasets
# Split: 80% training, 10% validation, 10% testing
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

# Display dataset statistics
num_examples = metadata.splits['train'].num_examples
num_classes = metadata.features['label'].num_classes
print(f"Total examples: {num_examples}")
print(f"Number of classes: {num_classes}")

# Create batched datasets for efficient training
BATCH_SIZE = 32
train_batches = raw_train.shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = raw_validation.map(format_image).batch(BATCH_SIZE).prefetch(1)
test_batches = raw_test.map(format_image).batch(1)  # Batch size of 1 for individual predictions

# Verify the shape of our batched data
for image_batch, label_batch in train_batches.take(1):
    print(f"Image batch shape: {image_batch.shape}")

# ==========================================
# MODEL BUILDING (TRANSFER LEARNING)
# ==========================================
# Configure the feature extractor
# MobileNetV2 is optimized for mobile/edge devices with good accuracy-size tradeoff
module_selection = ("mobilenet_v2", 224, 1280)
handle_base, pixels, FV_SIZE = module_selection
MODULE_HANDLE = "https://tfhub.dev/google/tf2-preview/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {} and output dimension {}".format(MODULE_HANDLE, IMAGE_SIZE, FV_SIZE))

# Load pre-trained MobileNetV2 feature extractor from TensorFlow Hub
# trainable=False: We freeze the base model weights (transfer learning)
feature_extractor = hub.KerasLayer(MODULE_HANDLE,
                                   input_shape=IMAGE_SIZE + (3,),
                                   output_shape=[FV_SIZE],
                                   trainable=False)

print("Building model with", MODULE_HANDLE)

# Build the complete model:
# 1. Feature extractor (pre-trained MobileNetV2)
# 2. Dense classification layer (trainable)
model = tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.summary()

# Compile the model with Adam optimizer and sparse categorical crossentropy loss
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
# SAVE MODEL FOR CONVERSION
# ==========================================
CATS_VS_DOGS_SAVED_MODEL = "exp_saved_model"
tf.saved_model.save(model, CATS_VS_DOGS_SAVED_MODEL)
print(f"Model saved to: {CATS_VS_DOGS_SAVED_MODEL}")

# ==========================================
# MODEL CONVERSION WITH QUANTIZATION
# ==========================================
"""
QUANTIZATION OVERVIEW:
---------------------
Quantization reduces model size and increases inference speed by converting
high-precision (32-bit float) weights and activations to lower precision (8-bit int).

Three Models to Create:
1. MODEL 1 (Baseline): No optimization
   - Uses 32-bit floating point
   - Largest size, highest accuracy, slowest inference
   
2. MODEL 2 (Dynamic Range Quantization): 
   - Weights quantized to 8-bit integers at rest
   - Activations remain as floats during inference
   - ~4x smaller size, slightly faster, minimal accuracy loss
   
3. MODEL 3 (Full Integer Quantization):
   - Both weights AND activations use 8-bit integers
   - Requires representative dataset for calibration
   - Smallest size, fastest on integer-only hardware, slight accuracy trade-off
"""

import pathlib
converter = tf.lite.TFLiteConverter.from_saved_model(CATS_VS_DOGS_SAVED_MODEL)

# ==========================================
# OPTION 1: NO OPTIMIZATION (MODEL 1)
# ==========================================
# Comment out the optimization sections below to create Model 1
# This creates a baseline model with no compression

# ==========================================
# OPTION 2: DYNAMIC RANGE QUANTIZATION (MODEL 2)
# ==========================================
# Uncomment this line for Model 2
# This enables post-training quantization using default optimizations
# Converts weights from float32 to int8, reducing model size by ~4x
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# ==========================================
# OPTION 3: FULL INTEGER QUANTIZATION (MODEL 3)
# ==========================================
# Uncomment ALL the following lines for Model 3
# This requires both the optimization flag AND a representative dataset

# STEP 1: Enable default optimizations
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# STEP 2: Provide representative dataset for calibration
# The converter uses this data to determine the optimal quantization parameters
# for activations (input/output ranges)
# def representative_data_gen():
#     """
#     Generator function that yields sample inputs from test dataset.
#     The converter uses these samples to calibrate activation quantization.
#     100 samples is typically sufficient for good calibration.
#     """
#     for input_value, _ in test_batches.take(100):
#         yield [input_value]

# converter.representative_dataset = representative_data_gen

# STEP 3: Enforce integer-only operations
# This ensures the model uses only int8 operations (no fallback to float)
# Required for maximum performance on integer-only accelerators
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# ==========================================
# CONVERT AND SAVE
# ==========================================
tflite_model = converter.convert()
tflite_models_dir = pathlib.Path("/tmp/")

# Change the filename here depending on which model you're creating:
# - model1.tflite (no optimization)
# - model2.tflite (dynamic range quantization)
# - model3.tflite (full integer quantization)
tflite_model_file = tflite_models_dir / 'model1.tflite'
tflite_model_file.write_bytes(tflite_model)

# Display file size
file_size_mb = tflite_model_file.stat().st_size / (1024 * 1024)
print(f"\nModel saved to: {tflite_model_file}")
print(f"File size: {file_size_mb:.2f} MB ({tflite_model_file.stat().st_size} bytes)")

"""
EXPECTED FILE SIZES:
-------------------
Without any optimizations:
  ~8.86 MB (8,857,848 bytes) - model1.tflite
  
With .optimizations property set (dynamic range quantization):
  ~2.63 MB (2,629,648 bytes) - model2.tflite
  ~70% size reduction!
  
With .optimizations + representative dataset (full integer quantization):
  ~2.84 MB (2,835,952 bytes) - model3.tflite
  Slightly larger due to additional metadata for int8 ops
"""

# ==========================================
# MODEL EVALUATION
# ==========================================
print("\n" + "=" * 60)
print("EVALUATING TFLITE MODEL")
print("=" * 60)

# Change the filename here to test different models
tflite_model_file = '/tmp/model1.tflite'

# Initialize the TFLite interpreter
# This is a lightweight runtime for executing TFLite models
interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
interpreter.allocate_tensors()

# Get input and output tensor indices
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

predictions = []
test_labels, test_imgs = [], []

# Run inference on 100 test images and measure speed
print("\nRunning predictions...")
for img, label in tqdm(test_batches.take(100), desc="Progress"):
    # Set input tensor
    interpreter.set_tensor(input_index, img)
    # Run inference
    interpreter.invoke()
    # Get output predictions
    predictions.append(interpreter.get_tensor(output_index))
    
    test_labels.append(label.numpy()[0])
    test_imgs.append(img)

"""
EXPECTED INFERENCE SPEEDS:
-------------------------
Model 1: ~32 it/s (iterations per second)
  - Fastest in Colab but slowest on actual edge devices
  - Uses optimized float32 operations
  
Model 2: ~16 it/s
  - Moderate speed in Colab
  - Faster on mobile devices due to smaller size
  
Model 3: ~1.2 it/s
  - Slowest in Colab (not optimized for int8)
  - FASTEST on dedicated edge hardware with int8 accelerators
  
Note: Speeds vary based on hardware. TFLite is optimized for mobile/edge,
      not Colab VMs, so Model 3 appears slower here but excels on real devices.
"""

# Calculate accuracy
score = 0
for item in range(0, 100):
    prediction = np.argmax(predictions[item])
    label = test_labels[item]
    if prediction == label:
        score = score + 1

print(f"\nAccuracy: {score}/100 correct predictions ({score}%)")

"""
EXPECTED ACCURACY:
-----------------
Model 1: ~100% correct (99-100/100)
  - Full precision, highest accuracy
  
Model 2: ~99% correct (98-100/100)
  - Minimal accuracy loss from quantization
  
Model 3: ~99% correct (98-100/100)
  - Similar to Model 2, slight variations due to int8 operations
  
Note: Small variations (±1-2%) are normal due to random initialization
"""

# ==========================================
# VISUALIZATION
# ==========================================
class_names = ['cat', 'dog']

def plot_image(i, predictions_array, true_label, img):
    """
    Plots a single prediction with color-coded accuracy:
    - Green: Correct prediction
    - Red: Incorrect prediction
    Shows confidence percentage and true label
    """
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    img = np.squeeze(img)
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    
    # Color code: green if correct, red if incorrect
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)

# Visualize the first N predictions
max_index = 15  # Adjust this to see more or fewer predictions
print(f"\nDisplaying {max_index} predictions...")

for index in range(0, max_index):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(index, predictions, test_labels, test_imgs)
    plt.show()

# ==========================================
# ASSIGNMENT INSTRUCTIONS
# ==========================================
"""
ASSIGNMENT TASKS:
================

1. CREATE MODEL 1 (Baseline):
   - Keep all quantization code commented out
   - Run the script
   - Note: file size (~8.86 MB), inference speed, accuracy

2. CREATE MODEL 2 (Dynamic Range Quantization):
   - Uncomment ONLY this line:
     converter.optimizations = [tf.lite.Optimize.DEFAULT]
   - Change filename to 'model2.tflite' in TWO places:
     * Line with: tflite_model_file = tflite_models_dir / 'model2.tflite'
     * Line with: tflite_model_file = '/tmp/model2.tflite'
   - Run the script
   - Note: file size (~2.63 MB), inference speed, accuracy
   - Compare with Model 1

3. CREATE MODEL 3 (Full Integer Quantization):
   - Uncomment ALL quantization code:
     * converter.optimizations = [tf.lite.Optimize.DEFAULT]
     * The entire representative_data_gen() function
     * converter.representative_dataset = representative_data_gen
     * converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
   - Change filename to 'model3.tflite' in TWO places
   - Run the script
   - Note: file size (~2.84 MB), inference speed, accuracy
   - Compare with Models 1 and 2

4. ANALYSIS QUESTIONS:
   - What is the size reduction percentage from Model 1 to Model 2?
   - How does accuracy change across the three models?
   - Why is Model 3 slower in Colab but faster on edge devices?
   - When would you choose each model type for deployment?
   - What trade-offs exist between size, speed, and accuracy?

5. BONUS EXPERIMENTS:
   - Try different numbers of representative samples (50, 200, 500)
   - Experiment with different batch sizes
   - Try on a different dataset (horses_or_humans)
   - Visualize which predictions each model gets wrong
"""

print("\n" + "=" * 60)
print("Assignment complete! Review the inline comments for analysis.")
print("=" * 60)
