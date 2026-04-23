"""
Linear Regression with TensorFlow — TinyML Assignment
=====================================================
A simple hands-on exercise: fit a line to data using a single-layer neural network (linear regression) in TensorFlow.

Key concepts:
- Model definition and compilation
- Loss function selection
- Training and visualizing learning progress
"""

import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# ── Sanity checks ─────────────────────────────────────────────────────────────
if tf.__version__.split('.')[0] != '2':
    raise Exception((f"The script is developed and tested for tensorflow 2. "
                     f"Current version: {tf.__version__}"))

if sys.version_info.major < 3:
    raise Exception((f"The script is developed and tested for Python 3. "
                     f"Current version: {sys.version_info.major}"))

# ── Data ──────────────────────────────────────────────────────────────────────
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# ── Model definition ──────────────────────────────────────────────────────────
# One dense layer, one neuron, input shape (1,)
SHAPE = (1,)
LOSS = 'mean_squared_error'

model = Sequential([Dense(units=1, input_shape=SHAPE)])
model.compile(optimizer='sgd', loss=LOSS)

# ── Training with callback to record predictions ──────────────────────────────
predictions = []
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        predictions.append(model.predict(xs))

callbacks = myCallback()

model.fit(xs, ys, epochs=300, callbacks=[callbacks], verbose=2)

# ── Plotting results ──────────────────────────────────────────────────────────
EPOCH_NUMBERS = [1, 25, 50, 150, 300]
plt.plot(xs, ys, label="Ys")
for EPOCH in EPOCH_NUMBERS:
    plt.plot(xs, predictions[EPOCH-1], label=f"Epoch = {EPOCH}")
plt.legend()
plt.show()
