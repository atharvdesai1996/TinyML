# Categorical Classification — Bean Disease Classifier

## Overview
This assignment builds a multi-class image classifier to identify bean leaf diseases. Unlike the binary (Horses vs Humans) classifier that used **binary crossentropy** and a single sigmoid output neuron, this problem has **3 classes** (healthy, bean rust, angular leaf spots), so we use **categorical crossentropy** and a **softmax** output layer.

## Why Categorical Crossentropy?

### Binary vs Categorical
| | Binary Classification | Categorical Classification |
|---|---|---|
| **Classes** | 2 (e.g., horse/human) | 3+ (e.g., healthy/rust/spots) |
| **Output layer** | `Dense(1, activation='sigmoid')` | `Dense(N, activation='softmax')` |
| **Loss function** | `binary_crossentropy` | `categorical_crossentropy` |
| **class_mode** | `'binary'` | `'categorical'` |
| **Output format** | Single probability (0–1) | Probability distribution over N classes |

### How Categorical Crossentropy Works
- The output layer has one neuron per class, using **softmax** activation which ensures all outputs sum to 1 (i.e., a probability distribution).
- **Categorical crossentropy** measures the difference between the predicted probability distribution and the true one-hot encoded label.
- Formula: `L = -Σ y_true * log(y_pred)` across all classes.

### Why Not Binary Crossentropy Here?
With 3 classes, a single sigmoid neuron can only output one value between 0 and 1 — it cannot distinguish between 3 categories. Softmax with categorical crossentropy naturally extends to any number of classes.

## Dataset
- **Source**: Bean disease images from Uganda (224×224 color images)
- **Classes**: Healthy, Bean Rust, Angular Leaf Spots
- **Splits**: Train, Validation, Test

## Model Architecture
```
Conv2D(16) → MaxPool → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → Flatten → Dense(512, relu) → Dense(3, softmax)
```

## Key Choices
- **Optimizer**: Adam (adaptive learning rate)
- **Loss**: `categorical_crossentropy`
- **Epochs**: 20
- **Data Augmentation**: rotation, shifts, shear, zoom, horizontal flip

## How to Run
1. Download the bean dataset (see download commands in script).
2. Extract to `/tmp/train`, `/tmp/validation`, `/tmp/test`.
3. Run `bean_disease_classifier.py`.

## Files
- `bean_disease_classifier.py` — Full training script with augmentation and accuracy plotting.
