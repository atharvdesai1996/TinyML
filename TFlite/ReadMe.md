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
