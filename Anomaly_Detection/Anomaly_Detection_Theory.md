# Anomaly Detection — Background Theory

## 1. What is Anomaly Detection?

Anomaly detection is the task of identifying data points that differ significantly from the "normal" distribution. It shows up everywhere in TinyML:

- Predictive maintenance on motors and pumps
- Health monitoring (ECG, EEG, respiration)
- Structural health monitoring
- Fraud & intrusion detection

The characteristic challenge: **anomalies are rare, diverse, and often unlabeled**. Collecting representative examples of every possible failure mode is impractical, so we cannot rely on standard supervised classification.

---

## 2. Why Unsupervised?

In real deployments:

- Failures are expensive to induce, so labeled anomaly data is scarce.
- New failure modes may appear that weren't in the training set.
- Labeling requires expert knowledge (a cardiologist, a mechanical engineer, etc.).

Therefore we train **only on (mostly) normal data** and treat "unusual" as anything the model can't reproduce well.

In this project, we deliberately contaminate the training set with 10% anomalies to simulate the realistic case where labels aren't perfectly clean.

---

## 3. Autoencoders

An **autoencoder** is a neural network that learns to reconstruct its input through a bottleneck:

```
    input --> Encoder --> z (latent code) --> Decoder --> reconstruction
     (140)                (very small)                       (140)
```

- The **encoder** compresses the input into a low-dimensional latent vector `z`.
- The **decoder** attempts to reconstruct the original input from `z`.
- Trained to minimize a reconstruction loss (MAE or MSE) between input and output.

Because information must flow through the narrow bottleneck, the network can only afford to memorize the *statistical structure of the majority class* — normal rhythms.

### Anomaly scoring

At inference time we compute:

```
score(x) = || x - decoder(encoder(x)) ||
```

- Low score -> looks like a normal training example.
- High score -> poorly reconstructed -> likely anomaly.

Choose a **threshold** on this score to make the final normal/anomaly decision.

---

## 4. Choosing the Bottleneck Size

This is the single most important architectural knob.

- **Too large** -> the autoencoder has enough capacity to reconstruct even the contaminating anomalies. The gap between normal and anomalous reconstruction error shrinks, and AUC drops.
- **Too small** -> even normal rhythms can't be reconstructed accurately. Everything looks abnormal; precision suffers.

For ECG5000 with a Dense(8) preceding layer, an **embedding of size 2** hits the sweet spot.

Guideline: sweep the bottleneck size and pick whichever maximizes **AUC on a validation set**.

---

## 5. Evaluation: ROC and AUC

The **ROC curve** plots the True Positive Rate against the False Positive Rate as the classification threshold is swept.

- **AUC (Area Under the ROC Curve)** summarizes the curve as a single number in [0, 1].
- AUC = 0.5 -> random classifier.
- AUC -> 1.0 -> perfect ranking of normals above anomalies.

AUC is **threshold-independent**: it tells us how good our reconstruction-error *scoring* is, regardless of where we later draw the line. That makes it ideal for choosing the model architecture (the embedding size in our case).

---

## 6. Choosing an Operating Threshold

Once we've picked a model, we still have to pick a scalar threshold on the reconstruction error to make hard decisions.

Standard metrics:

- **Accuracy** — fraction of correct predictions.
- **Precision** — of all rhythms we flagged as *normal*, how many really were?
- **Recall** — of all truly *normal* rhythms, how many did we correctly keep?

There's usually a **precision/recall trade-off**. The right operating point depends on the application:

- For ECG monitoring, missing a dangerous rhythm (false negative on the anomaly class) is worse than raising a false alarm — so we often move off the pure "knee" of the ROC curve toward higher recall for anomalies.

A good starting point is the **knee of the ROC curve**, then adjust based on the cost of each error type.

---

## 7. Why Autoencoders Fit TinyML

- **Small footprint** — the model here is only a handful of Dense layers; it easily fits on a microcontroller.
- **No labels needed at deploy time** — great for edge sensors.
- **On-device inference** avoids streaming raw sensor data over the network, which is often infeasible due to bandwidth and power.
- **Trade-off awareness** matters: false alarms waste power and human attention, missed anomalies can be catastrophic. Threshold selection is an engineering decision, not just a stats one.

Note: autoencoders for anomaly detection tend to have **low transferability** — each deployment target usually needs its own training on data from its own sensor and environment.

---

## 8. Alternatives Worth Knowing

- **K-Means clustering** — classical, easy to implement, but scales poorly and struggles with high-dimensional signals.
- **One-class SVM / Isolation Forest** — good classical baselines.
- **Variational autoencoders (VAEs)** — probabilistic reconstruction, often more robust.
- **LSTM / temporal-conv autoencoders** — better for long time-series where order matters more than we assume here.

For ECG5000 the vanilla dense autoencoder is more than enough.

---

## 9. Further Reading

- Chandola, Banerjee, Kumar — *Anomaly Detection: A Survey* (ACM Computing Surveys, 2009)
- Goodfellow, Bengio, Courville — *Deep Learning*, Chapter 14 (Autoencoders)
- TensorFlow tutorial: <https://www.tensorflow.org/tutorials/generative/autoencoder>
- ECG5000 description: <http://www.timeseriesclassification.com/description.php?Dataset=ECG5000>
