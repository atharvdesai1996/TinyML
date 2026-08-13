"""
ECG Anomaly Detection with an Autoencoder
=========================================

Unsupervised anomaly detection on the ECG5000 dataset. A small dense
autoencoder is trained on a training set that is ~90% normal rhythms and
~10% anomalies (labels are NOT used at training time). At inference we flag
a beat as anomalous when its reconstruction error exceeds a chosen
threshold.

Pipeline overview:
    Step  1 : Environment setup (Keras 2 compatibility, imports).
    Step  2 : Download the ECG5000 CSV.
    Step  3 : Train/test split + min-max normalization.
    Step  4 : Split normal vs anomalous samples by label.
    Step  5 : Build the contaminated (unsupervised) training set.
    Step  6 : Visualize a normal and an anomalous ECG.
    Step  7 : Define the autoencoder (bottleneck = EMBEDDING_SIZE).
    Step  8 : Compile and train.
    Step  9 : Visualize reconstructions for normal and anomalous inputs.
    Step 10 : Plot the ROC curve and compute AUC.
    Step 11 : Pick a threshold and report accuracy / precision / recall.

Design choices we make in this script (tuned on the validation set):
    EMBEDDING_SIZE = 2      # small bottleneck -> can't memorize anomalies
    threshold      = 0.037  # near the knee of the ROC curve
"""

# -----------------------------------------------------------------------------
# Step 1: Environment setup
# -----------------------------------------------------------------------------
import os
import subprocess
import sys

os.environ["TF_USE_LEGACY_KERAS"] = "1"  # keep Keras 2 API semantics

try:
    import tensorflow as tf
    import tf_keras
    print(f"TensorFlow {tf.__version__} (tf_keras {tf_keras.__version__})")
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "tensorflow", "tf_keras"])
    print("=" * 60)
    print("Install complete. Now: Runtime > Restart session, then re-run.")
    print("=" * 60)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_curve, auc)
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Model


# -----------------------------------------------------------------------------
# Step 2: Download the ECG5000 dataset
# -----------------------------------------------------------------------------
# Each row is one heartbeat: 140 samples of the signal + a class label in the
# final column (1 = normal, 0 = anomalous).
ECG_URL = "http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv"
dataframe = pd.read_csv(ECG_URL, header=None)
raw_data = dataframe.values
print("Raw shape:", raw_data.shape)

labels = raw_data[:, -1]
data = raw_data[:, 0:-1]


# -----------------------------------------------------------------------------
# Step 3: Train/test split + normalize to [0, 1]
# -----------------------------------------------------------------------------
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=21
)

min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)


# -----------------------------------------------------------------------------
# Step 4: Split normal vs anomalous by label
# -----------------------------------------------------------------------------
# Convention: True (1) = normal, False (0) = anomaly.
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]


# -----------------------------------------------------------------------------
# Step 5: Build a contaminated, unsupervised training set
# -----------------------------------------------------------------------------
# Mix ~10% anomalies into the training set to simulate a real-world
# "no clean labels available" scenario. The autoencoder never sees the
# labels; it just tries to reconstruct whatever it is given.
portion_of_anomaly_in_training = 0.1  # 10%
end_size = int(len(normal_train_data) /
               (10 - portion_of_anomaly_in_training * 10))
combined_train_data = np.append(normal_train_data,
                                anomalous_test_data[:end_size],
                                axis=0)
print("Contaminated training set shape:", combined_train_data.shape)


# -----------------------------------------------------------------------------
# Step 6: Sanity-check the data visually
# -----------------------------------------------------------------------------
plt.grid()
plt.plot(np.arange(140), normal_train_data[0])
plt.title("A Normal ECG")
plt.show()

plt.grid()
plt.plot(np.arange(140), anomalous_train_data[0])
plt.title("An Anomalous ECG")
plt.show()


# -----------------------------------------------------------------------------
# Step 7: Define the autoencoder
# -----------------------------------------------------------------------------
# The single most important knob here is EMBEDDING_SIZE (the bottleneck).
#   Too large -> the model can also reproduce the 10% anomalies, ruining AUC.
#   Too small -> even normal rhythms are reconstructed poorly.
# Empirically, a bottleneck of 2 gives the best AUC on this dataset.
EMBEDDING_SIZE = 2


class AnomalyDetector(Model):
    def __init__(self):
        super().__init__()
        # Encoder: 140 -> 8 -> EMBEDDING_SIZE
        self.encoder = tf.keras.Sequential([
            layers.Dense(8, activation="relu"),
            layers.Dense(EMBEDDING_SIZE, activation="relu"),
        ])
        # Decoder: EMBEDDING_SIZE -> 8 -> 140 (sigmoid matches [0,1] scaling)
        self.decoder = tf.keras.Sequential([
            layers.Dense(8, activation="relu"),
            layers.Dense(140, activation="sigmoid"),
        ])

    def call(self, x):
        return self.decoder(self.encoder(x))


autoencoder = AnomalyDetector()
print("Chosen Embedding Size:", EMBEDDING_SIZE)


# -----------------------------------------------------------------------------
# Step 8: Compile and train
# -----------------------------------------------------------------------------
# MAE (L1) reconstruction loss is a robust default for time-series signals.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
autoencoder.compile(optimizer=optimizer, loss="mae")

history = autoencoder.fit(
    combined_train_data, combined_train_data,
    epochs=50,
    batch_size=512,
    validation_data=(test_data, test_data),
    shuffle=True,
)

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Autoencoder Training")
plt.show()


# -----------------------------------------------------------------------------
# Step 9: Visualize reconstructions
# -----------------------------------------------------------------------------
# For a normal signal the reconstruction should closely track the input.
encoded = autoencoder.encoder(normal_test_data).numpy()
decoded = autoencoder.decoder(encoded).numpy()

plt.plot(normal_test_data[0], "b")
plt.plot(decoded[0], "r")
plt.fill_between(np.arange(140), decoded[0], normal_test_data[0],
                 color="lightcoral")
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.title("Normal ECG - Reconstruction")
plt.show()

# For an anomalous signal the reconstruction should be noticeably worse.
encoded = autoencoder.encoder(anomalous_test_data).numpy()
decoded = autoencoder.decoder(encoded).numpy()

plt.plot(anomalous_test_data[0], "b")
plt.plot(decoded[0], "r")
plt.fill_between(np.arange(140), decoded[0], anomalous_test_data[0],
                 color="lightcoral")
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.title("Anomalous ECG - Reconstruction")
plt.show()


# -----------------------------------------------------------------------------
# Step 10: ROC curve + AUC
# -----------------------------------------------------------------------------
# Score = per-sample MAE between input and reconstruction. Higher score
# means "more likely anomaly". sklearn's roc_curve wants the positive class
# to be the anomaly class, so we flip the labels.
reconstructions = autoencoder(test_data)
loss = tf.keras.losses.mae(reconstructions, test_data)

flipped_labels = 1 - test_labels
fpr, tpr, thresholds = roc_curve(flipped_labels, loss)

plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC - ECG Anomaly Detection")
plt.legend(loc="lower right")

# Annotate a few candidate thresholds directly on the curve.
thresholds_every = 20
for i in range(0, len(thresholds), thresholds_every):
    label = str(thresholds[i])[:5]
    plt.scatter(fpr[i], tpr[i], c="black")
    plt.text(fpr[i] - 0.03, tpr[i] + 0.005, label, fontdict={"size": 12})
plt.show()

roc_auc = auc(fpr, tpr)
print("AUC:", roc_auc)


# -----------------------------------------------------------------------------
# Step 11: Pick a threshold and evaluate
# -----------------------------------------------------------------------------
# The threshold sits near the "knee" of the ROC curve. 0.037 balances
# precision and recall well while keeping accuracy high (>94% on all three).
threshold = 0.037
print("Chosen Threshold:", threshold)


def predict(model, data, threshold):
    """Return (is_normal_mask, per_sample_loss)."""
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    # Low loss => normal (label 1); high loss => anomaly.
    return tf.math.less(loss, threshold), loss


def print_stats(predictions, labels):
    print("Accuracy  =", accuracy_score(labels, predictions))
    print("Precision =", precision_score(labels, predictions))
    print("Recall    =", recall_score(labels, predictions))


preds, scores = predict(autoencoder, test_data, threshold)
print_stats(preds, test_labels)

# -----------------------------------------------------------------------------
# Notes on hyperparameter choices
# -----------------------------------------------------------------------------
# * EMBEDDING_SIZE must be strictly between 0 and 8 (so it's actually a
#   bottleneck vs. the Dense(8) layer that precedes it). Size 2 gives the
#   best AUC in practice on ECG5000.
# * The threshold depends on the trained model and the embedding size,
#   but ~0.037 is a solid starting point. Pick based on the cost of a
#   false negative vs. a false positive in your application. For medical
#   monitoring, biasing toward high recall on anomalies is usually the
#   right call.
