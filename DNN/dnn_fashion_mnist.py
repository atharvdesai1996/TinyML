"""
DNN Learning with TensorFlow - Fashion MNIST
=============================================
Exploring how Deep Neural Networks learn using the Fashion MNIST dataset.
Based on the TinyML / Intro to ML course (edX / Google).

Key concepts explored:
- Data normalization
- Effect of neuron count
- Effect of additional layers
- Early stopping with custom callbacks
"""

import sys
import tensorflow as tf

# ── Sanity checks ─────────────────────────────────────────────────────────────

if sys.version_info.major < 3:
    raise Exception(
        f"This script requires Python 3. Current version: {sys.version_info.major}"
    )

if tf.__version__.split(".")[0] != "2":
    raise Exception(
        f"This script requires TensorFlow 2. Current version: {tf.__version__}"
    )

# ── 1. Load & normalise data ──────────────────────────────────────────────────

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Pixel values are in [0, 255]. Normalising to [0, 1] helps the network learn
# faster and more stably — large input values can push activations into
# saturation and make gradients vanish or explode.
training_images = training_images / 255.0
test_images     = test_images     / 255.0

# ── 2. Baseline model (512 neurons, 1 hidden layer) ───────────────────────────

print("\n=== Baseline model (512 neurons) ===")

baseline_model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(10,  activation="softmax"),
])

baseline_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

baseline_model.fit(training_images, training_labels, epochs=5)
baseline_model.evaluate(test_images, test_labels)

# ── 3. Wider model (1 024 neurons) ────────────────────────────────────────────
# Doubling the neurons gives the model more capacity to find patterns, but also
# increases training time roughly proportionally. Accuracy gains are often
# marginal on a simple dataset like this.

print("\n=== Wider model (1 024 neurons) ===")

NUMBER_OF_NEURONS = 1024

wider_model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(NUMBER_OF_NEURONS, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

wider_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

wider_model.fit(training_images, training_labels, epochs=5)
wider_model.evaluate(test_images, test_labels)

# ── 4. Deeper model (extra hidden layer) ──────────────────────────────────────
# Adding depth lets the network learn hierarchical representations.
# On this simple dataset the gain is modest, but the principle scales to
# much harder problems (e.g. image recognition, NLP).

print("\n=== Deeper model (2 x 512 hidden layers) ===")

deeper_model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(512, activation="relu"),   # extra layer
    tf.keras.layers.Dense(10,  activation="softmax"),
])

deeper_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

deeper_model.fit(training_images, training_labels, epochs=5)
deeper_model.evaluate(test_images, test_labels)

# ── 5. Non-normalised data (to show why normalisation matters) ─────────────────
# Training on raw [0, 255] pixel values is noticeably slower and less accurate.
# Large activations cause unstable gradients during back-propagation.

print("\n=== Deeper model trained on NON-normalised data ===")

training_images_non = training_images * 255.0
test_images_non     = test_images     * 255.0

non_norm_model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(10,  activation="softmax"),
])

non_norm_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

non_norm_model.fit(training_images_non, training_labels, epochs=5)
non_norm_model.evaluate(test_images_non, test_labels)

# ── 6. Early-stopping callback ────────────────────────────────────────────────
# Instead of always training for a fixed number of epochs, a callback lets us
# stop as soon as progress stalls — saving time and preventing overfitting.

class EarlyStopOnPlateau(tf.keras.callbacks.Callback):
    """Stop training when accuracy improvement drops below a threshold.

    The check only kicks in after the first epoch so we always complete at
    least two epochs of training.
    """

    def __init__(self, min_delta: float = 1e-4):
        super().__init__()
        self.min_delta = min_delta
        self.prev_acc  = None

    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get("accuracy")

        if self.prev_acc is not None:
            improvement = abs(acc - self.prev_acc)
            if improvement < self.min_delta:
                print(
                    f"\nEpoch {epoch + 1}: accuracy change ({improvement:.6f}) "
                    f"< threshold ({self.min_delta}). Stopping early."
                )
                self.model.stop_training = True

        self.prev_acc = acc


print("\n=== Deeper model with early-stop callback ===")

callback_model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(10,  activation="softmax"),
])

callback_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callback_model.fit(
    training_images,
    training_labels,
    epochs=20,                           # high ceiling; callback will cut it short
    callbacks=[EarlyStopOnPlateau(min_delta=1e-4)],
)

callback_model.evaluate(test_images, test_labels)
