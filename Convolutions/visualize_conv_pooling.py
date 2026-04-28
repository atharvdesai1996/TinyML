import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models

# Assume model, val_images, val_labels are already loaded and preprocessed
# model: trained CNN model
# val_images: validation images, shape (num_samples, 28, 28, 1)
# val_labels: validation labels

def show_image(img_idx):
    plt.figure()
    plt.imshow(val_images[img_idx].reshape(28, 28), cmap='gray')
    plt.grid(False)
    plt.title(f"Label: {val_labels[img_idx]}")
    plt.show()

# Print first 100 labels for reference
print(val_labels[:100])

f, axarr = plt.subplots(3, 2, figsize=(8, 10))
FIRST_IMAGE = 0
SECOND_IMAGE = 23
THIRD_IMAGE = 28
CONVOLUTION_NUMBER = 1  # Visualize the 2nd filter

layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.inputs, outputs=layer_outputs)

for x in range(0, 2):  # 0: first conv layer, 1: first maxpool layer
    f1 = activation_model.predict(val_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[0, x].grid(False)
    f2 = activation_model.predict(val_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[1, x].grid(False)
    f3 = activation_model.predict(val_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[2, x].grid(False)

plt.suptitle('Convolution and Pooling Visualizations (cmap="inferno")')
plt.show()

show_image(FIRST_IMAGE)
show_image(SECOND_IMAGE)
show_image(THIRD_IMAGE)
