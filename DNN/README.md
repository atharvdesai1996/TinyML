# DNN Learning with TensorFlow — Fashion MNIST

A hands-on exploration of how **Deep Neural Networks (DNNs)** learn, using the
classic [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.

---

## What is a DNN?

A **Deep Neural Network** is a stack of layers of artificial neurons. Each layer
learns to recognise increasingly abstract features in the data:

```
Input  →  [Hidden layer 1]  →  [Hidden layer 2]  →  ...  →  Output
```

Each neuron computes a weighted sum of its inputs, applies an **activation
function** (e.g. ReLU), and passes the result forward. During training,
**back-propagation** adjusts the weights to minimise a loss function.

---

## What is ReLU?

**ReLU** stands for **Rectified Linear Unit**. It is the most common activation function in modern neural networks.

**Definition:**

    f(x) = max(0, x)

This means:
- Negative values become 0
- Positive values stay the same

**Why use ReLU?**
- It introduces non-linearity, allowing the network to learn complex patterns (not just lines).
- It acts as a signal filter: only strong (positive) signals pass through, weak/negative ones are blocked.

**Sensor analogy:**
Imagine each neuron is a sensor. If the signal is negative (weak or noise), ReLU blocks it (outputs 0). If the signal is positive (strong), it passes through. This makes the network focus on useful features and ignore noise.

**In code:**
```python
Dense(20, activation=tf.nn.relu)
```
Each neuron's output is passed through ReLU before moving to the next layer.

**Limitation:**
If a neuron always outputs negative values, it will always be zero after ReLU (a "dead neuron").

**Bottom line:**
ReLU makes learning efficient, keeps only useful signals, and enables deep networks to learn complex data.

---

## Dataset — Fashion MNIST

- **60 000** training images, **10 000** test images
- Greyscale, **28 × 28 pixels**
- **10 classes**: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt,
  Sneaker, Bag, Ankle boot

---

## Experiments

### 1 — Baseline model
A single hidden layer with **512 neurons**.

> Achieved ~89 % training accuracy and ~88 % test accuracy after 5 epochs.

### 2 — Wider model (more neurons)
Doubling the hidden layer to **1 024 neurons** gives the model more capacity.

> Result: marginal accuracy improvement, but roughly **2× longer** training time
> per epoch. Diminishing returns on a simple dataset.

### 3 — Deeper model (more layers)
Adding a second hidden layer (2 × 512) instead of just more neurons.

> Depth allows the network to learn **hierarchical features**. On Fashion MNIST
> the gain is small, but depth is what powers CNNs and Transformers on hard tasks.

### 4 — Effect of normalisation
Training without normalising pixel values (raw 0–255 range).

> Accuracy drops noticeably (~84 % vs ~89 %) and training is slower. Large inputs
> push ReLU activations into unstable regions, making gradients harder to
> propagate correctly.

### 5 — Custom early-stopping callback
Instead of a fixed epoch count, a `tf.keras.callbacks.Callback` monitors
accuracy each epoch and stops training when improvement falls below a threshold.

> Saves compute and reduces the risk of overfitting — useful when you don't know
> in advance how many epochs are needed.

---

## Key takeaways

| Concept | What changes | Why it matters |
|---|---|---|
| **Normalisation** | Scale inputs to [0, 1] | Stable gradients, faster convergence |
| **Width** (neurons) | More neurons per layer | Higher capacity, more compute |
| **Depth** (layers) | More layers | Hierarchical feature learning |
| **Callbacks** | Stop/adjust training dynamically | Efficiency, prevent overfitting |

---

## How to run

```bash
pip install tensorflow
python dnn_fashion_mnist.py
```

Requires **Python 3** and **TensorFlow 2**.

---

## Structure

```
DNN/
├── dnn_fashion_mnist.py   # All experiments in a single runnable script
└── README.md              # This file
```
