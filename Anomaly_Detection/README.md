# ECG Anomaly Detection with Autoencoders

An unsupervised anomaly-detection project on the [ECG5000](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000) dataset. A small autoencoder is trained on a mostly-normal training set (with 10% anomalies mixed in) and abnormal heartbeats are flagged at inference time by thresholding the reconstruction error.

---

## Folder Contents

| File | Description |
|------|-------------|
| `README.md` | Project overview & instructions (this file) |
| `Anomaly_Detection_Theory.md` | Background theory: autoencoders, ROC/AUC, thresholding |
| `ecg_anomaly_detection.py` | Full training + evaluation script with step-by-step comments |

---

## Objective

Build an unsupervised ECG classifier that decides whether a heartbeat is **normal** or **anomalous**, using only reconstruction error — no labels are used during training.

Two design choices drive the results:

1. **Embedding size** of the autoencoder bottleneck — chosen to maximize **AUC**.
2. **Reconstruction-error threshold** — chosen to jointly maximize **accuracy, precision, and recall**.

> Convention: label `1` = normal rhythm, label `0` = anomaly.

---

## Dataset

ECG5000 (140-sample univariate time series per heartbeat), fetched at runtime from:

```
http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv
```

- 80 / 20 train/test split.
- Values min-max normalized to `[0, 1]`.
- Training set is **contaminated with ~10% anomalies** to simulate a realistic unsupervised setting where clean labels are unavailable.

---

## Workflow

1. **Load & normalize** the ECG signals.
2. **Split** into normal / anomalous subsets by label.
3. **Contaminate** the training set with 10% anomalies (labels discarded for training).
4. **Visualize** one normal and one anomalous heartbeat as a sanity check.
5. **Build a small autoencoder** (`Dense(8) -> Dense(EMBEDDING_SIZE) -> Dense(8) -> Dense(140)`).
6. **Train** with MAE loss and Adam (`lr=0.01`) for 50 epochs.
7. **Compare** input vs reconstruction for both normal and anomalous samples.
8. **Plot ROC** with threshold annotations and compute **AUC**.
9. **Pick a threshold** on the reconstruction error and report accuracy / precision / recall.

---

## How to Run

```bash
pip install tensorflow tf_keras scikit-learn pandas matplotlib
python ecg_anomaly_detection.py
```

Works locally or in Google Colab.

---

## Chosen Hyperparameters

After sweeping, the following gave the best trade-off:

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| `EMBEDDING_SIZE` | `2` | Small enough to prevent memorizing anomalies, large enough to encode normal rhythms |
| `threshold` | `~0.037` | Near the knee of the ROC curve |

Typical results at these settings:
- **AUC**: high 0.9x
- **Accuracy / Precision / Recall**: all > 94%

---

## Why This Works

The autoencoder is trained almost entirely on normal rhythms, so it learns a compact latent representation that reconstructs *normal* signals accurately. Anomalous signals fall outside the learned manifold and reconstruct poorly — their high MAE lets us detect them without ever having seen a label.

Keeping the bottleneck **very small (size 2)** is crucial: a larger latent space would let the model reconstruct the contaminating 10% anomalies too, collapsing the error gap we rely on.

See `Anomaly_Detection_Theory.md` for the theoretical background.
