# Linear Regression with TensorFlow — TinyML Assignment

A hands-on exercise to fit a straight line to data using a single-layer neural network (linear regression) in TensorFlow.

---

## Problem
Given:
- Inputs: `xs = [-1, 0, 1, 2, 3, 4]`
- Outputs: `ys = [-3, -1, 1, 3, 5, 7]`

The goal is to learn the relationship between `xs` and `ys` (which is `y = 2x - 1`).

---

## Approach
- **Model:** Single dense layer with 1 neuron (linear regression)
- **Loss:** Mean squared error
- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Callback:** Records predictions at every epoch for visualization

---

## Visualization
The script plots the model's predictions at different epochs (1, 25, 50, 150, 300) to show how the line fits the data over time.

---

## How to run

```bash
pip install tensorflow matplotlib numpy
python linear_regression_tf.py
```

---

## Files

```
LinearRegression/
├── linear_regression_tf.py   # The assignment script
└── README.md                 # This file
```
