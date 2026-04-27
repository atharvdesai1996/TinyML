"""
Convolutions and Pooling — TinyML Assignment
============================================
Explore how convolutions and pooling work on images, and visualize their effects.

- Demonstrates edge detection with custom filters (kernels)
- Shows how max pooling reduces image size while preserving features
"""

import cv2
import numpy as np
from scipy import datasets
import matplotlib.pyplot as plt

# Load the sample image (stairwell)
i = datasets.ascent().astype(np.int32)

# Show the original image
plt.figure(figsize=(6,6))
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)
plt.title('Original Image')
plt.show()

# Prepare for convolution
i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]

# Define a filter (kernel) for edge detection
# Try different filters for different effects
# Example: vertical edge detection
filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
weight = 1

# Apply convolution
for x in range(1, size_x-1):
    for y in range(1, size_y-1):
        convolution = 0.0
        convolution += i[x-1, y-1] * filter[0][0]
        convolution += i[x,   y-1] * filter[1][0]
        convolution += i[x+1, y-1] * filter[2][0]
        convolution += i[x-1, y]   * filter[0][1]
        convolution += i[x,   y]   * filter[1][1]
        convolution += i[x+1, y]   * filter[2][1]
        convolution += i[x-1, y+1] * filter[0][2]
        convolution += i[x,   y+1] * filter[1][2]
        convolution += i[x+1, y+1] * filter[2][2]
        convolution *= weight
        # Clamp to [0, 255]
        convolution = max(0, min(255, convolution))
        i_transformed[x, y] = convolution

# Show the convolved image
plt.figure(figsize=(6,6))
plt.gray()
plt.grid(False)
plt.imshow(i_transformed)
plt.title('After Convolution (Edge Detection)')
plt.show()

# Max Pooling (4x4)
new_x = int(size_x / 4)
new_y = int(size_y / 4)
newImage = np.zeros((new_x, new_y))
for x in range(0, size_x, 4):
    for y in range(0, size_y, 4):
        pixels = []
        for dx in range(4):
            for dy in range(4):
                if x+dx < size_x and y+dy < size_y:
                    pixels.append(i_transformed[x+dx, y+dy])
        pixels.sort(reverse=True)
        newImage[int(x/4), int(y/4)] = pixels[0]

# Show the pooled image
plt.figure(figsize=(6,6))
plt.gray()
plt.grid(False)
plt.imshow(newImage)
plt.title('After Max Pooling (4x4)')
plt.show()
