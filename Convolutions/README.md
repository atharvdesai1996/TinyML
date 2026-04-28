# Convolutions and Pooling — TinyML Assignment

![CNN Architecture for Fashion-MNIST Classification](cnn_architecture.png)

This assignment demonstrates the basics of **convolutions** and **pooling** in image processing, which are foundational concepts for Convolutional Neural Networks (CNNs).

---

## What are Convolutions?
A **convolution** is a mathematical operation that applies a filter (also called a kernel) to an image. The filter slides over the image, multiplying and summing pixel values to highlight specific features (like edges or lines). This helps neural networks learn patterns that are spatially local and translation-invariant.

- Example: Edge detection filters can highlight vertical or horizontal lines in an image.

## What is Pooling?
**Pooling** is a technique to reduce the spatial size of an image while preserving important features. The most common type is **max pooling**, which replaces a block of pixels with their maximum value. This reduces computation and helps the network focus on the most prominent features.

---

## What does the code do?
1. Loads a sample grayscale image (stairwell) from `scipy`.
2. Applies a 3×3 convolution filter to detect edges (you can try different filters for different effects).
3. Visualizes the result after convolution.
4. Applies 4×4 max pooling to compress the image, keeping only the strongest features.
5. Visualizes the pooled image.

---

## How to run

```bash
pip install numpy matplotlib scipy opencv-python
python convolutions_and_pooling.py
```

---

## Files
```
Convolutions/
├── convolutions_and_pooling.py   # The assignment script
└── README.md                     # This file
```

---

## Visualizing Convolutions and Pooling

The script `visualize_conv_pooling.py` lets you see what the convolution and pooling layers are actually doing to your images. It shows the activations (feature maps) for selected images and filters, helping you understand what features the model is learning.

- The script selects three images (e.g., all shoes) and visualizes the output of the first convolution and pooling layers for a specific filter.
- The output shape like `(1, 26, 26, 32)` means:
  - 1: batch size (one image)
  - 26 × 26: spatial size after convolution
  - 32: number of filters (feature maps)
- You can change `CONVOLUTION_NUMBER` to see different filters.

### What is `cmap='inferno'`?
- `cmap` stands for colormap. `'inferno'` is a visually striking color map that highlights differences in activation values, making patterns easier to see than plain grayscale.
- Brighter colors = higher activation (stronger feature detected).

### What is ReLU?
- **ReLU** (Rectified Linear Unit) is an activation function: `f(x) = max(0, x)`.
- It keeps only positive signals, making the network focus on strong features and ignore weak/negative ones.
- ReLU introduces non-linearity, allowing the network to learn complex patterns, not just straight lines.

---

**Try running the script and explore how different filters and layers highlight different features in your images!**
