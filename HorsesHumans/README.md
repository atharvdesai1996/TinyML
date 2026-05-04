# Horses vs Humans Binary Classifier

This assignment demonstrates how to build a binary image classifier using convolutional neural networks (CNNs) to distinguish between horses and humans. It covers data loading, model training, prediction, and layer visualization.

---

## Key Concepts

### Sigmoid Activation
- **Definition:** `sigmoid(x) = 1 / (1 + exp(-x))`
- **Use:** Outputs a value between 0 and 1, representing probability. Used for binary classification (e.g., horse vs human).
- **In the model:**
  ```python
  tf.keras.layers.Dense(1, activation='sigmoid')
  ```

### Binary Crossentropy Loss
- **Definition:** Measures the difference between true labels (0 or 1) and predicted probabilities.
- **Use:** Used for binary classification tasks with sigmoid output.
- **In the model:**
  ```python
  model.compile(loss='binary_crossentropy', ...)
  ```

### RMSprop Optimizer
- **Definition:** An adaptive learning rate optimizer that adjusts the learning rate for each parameter.
- **Use:** Helps models converge faster and more reliably, especially for image data.
- **In the model:**
  ```python
  optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
  model.compile(optimizer=optimizer, ...)
  ```

---

## Workflow
1. Download and extract data (horses and humans images)
2. Organize data using `ImageDataGenerator`
3. Build the CNN model with several Conv2D and Dense layers
4. Train the model using binary crossentropy loss and RMSprop optimizer
5. Evaluate and visualize accuracy
6. Visualize intermediate layer outputs to see what features the model is learning

---

## How to run
1. Download the dataset zips and place them in `/tmp/` (or let the script download them if running in Colab)
2. Run `horse_human_classifier.py`

---

## Files
```
HorsesHumans/
├── horse_human_classifier.py   # Main script
└── README.md                   # This file
```

# Overfitting and Data Augmentation in Binary Image Classification

## Overfitting
Overfitting occurs when a machine learning model learns the training data too well, including its noise and outliers, resulting in poor generalization to new, unseen data. In training curves, overfitting is often observed when training accuracy continues to improve while validation accuracy plateaus or decreases.

**Symptoms of Overfitting:**
- High training accuracy but low validation accuracy
- Large gap between training and validation loss

## Data Augmentation
Data augmentation is a technique used to artificially increase the diversity of the training dataset by applying random transformations (such as rotations, shifts, flips, etc.) to the input images. This helps the model generalize better and reduces overfitting.

**Common Augmentation Techniques:**
- Rotation
- Width/height shift
- Shear
- Zoom
- Horizontal/vertical flip

## Example: Horses vs Humans Classifier
The script `overfitting_and_augmentation.py` demonstrates both overfitting and the use of data augmentation. It uses Keras' `ImageDataGenerator` to apply augmentation and plots training/validation accuracy to visualize overfitting.

### Key Code Snippet
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True,
    #fill_mode='nearest'
)
```

### How to Run
1. Download the Horses vs Humans dataset and extract it to `/tmp/horse-or-human` and `/tmp/validation-horse-or-human`.
2. Run `overfitting_and_augmentation.py`.
3. Observe the accuracy plot to see the effect of augmentation on overfitting.

---

**Summary:**
- Overfitting is a common issue in deep learning, especially with small datasets.
- Data augmentation is an effective way to reduce overfitting and improve model generalization.
