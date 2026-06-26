# Cats vs. Dogs: Transfer Learning & TensorFlow Lite

This repository contains a complete pipeline for training an image classification model to distinguish between cats and dogs, and subsequently converting that model into a highly optimized format suitable for mobile and edge devices (TensorFlow Lite).

## Overview

The script leverages **Transfer Learning** using a pre-trained `MobileNetV2` model from TensorFlow Hub. By freezing the base layers and attaching a new classification head, we can achieve high accuracy (typically >98%) in just a few epochs without needing a massive dataset or extensive compute power.

Once trained, the model is exported as a standard `SavedModel` and then run through the `TFLiteConverter` to produce a lightweight `.tflite` file. Finally, the script uses the TFLite Interpreter to verify the converted model's accuracy on unseen test images.

## Features
* **Automated Dataset Handling:** Downloads and splits the Microsoft Cats vs. Dogs dataset using `tensorflow_datasets`.
* **Transfer Learning:** Utilizes Google's MobileNetV2 architecture.
* **TFLite Conversion:** Compresses the model for deployment on Android, iOS, or IoT devices (like Raspberry Pi).
* **Interpreter Verification:** Includes a built-in test loop to ensure the `.tflite` model maintains accuracy after conversion.

## Prerequisites

Ensure you have Python 3.8+ installed. You will need the following libraries to run the script:

```bash
pip install tensorflow==2.19.0 tensorflow-hub==0.16.0 tensorflow-datasets==4.9.9 numpy==2.0.2 protobuf==3.20.3 matplotlib tqdm
```

## Script explanation 
The number 1280 represents the Feature Vector Size (FV_SIZE) or output dimension of the pre-trained MobileNetV2 model.

Here is a breakdown of what 1280 actually means for the model:

1. The Output of the "Feature Extractor"
When you feed an image (which has 224x224 pixels and 3 color channels) into the MobileNetV2 base model, it passes through dozens of convolutional layers. By the time the image reaches the end of this pre-trained base, all of its visual information has been mathematically compressed and distilled into a single, 1-dimensional list of exactly 1,280 numbers.

2. What Do These 1,280 Numbers Represent?
These numbers are the features the model has learned to look for. While we humans see a picture of a cat, the MobileNetV2 model translates that picture into 1,280 different "scores." These scores represent various abstract visual patterns it detected in the image, such as:

The presence of pointy ears.
The texture of fur.
The shape of a snout or eyes.
Countless other complex geometrical patterns we can't easily name.
