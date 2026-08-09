# Transfer Learning — Background Theory

## 1. What is Transfer Learning?

**Transfer learning** is the practice of taking a model trained on one task (the *source task*) and reusing it — in whole or in part — to solve a different but related task (the *target task*).

The core insight: deep neural networks trained on large, diverse datasets (e.g. ImageNet, with 1.4M images across 1000 classes) learn **hierarchical feature representations** that generalize well beyond the original classification task. Early layers learn generic features (edges, textures, color blobs), middle layers learn parts (eyes, wheels, fur patterns), and only the final layers become highly task-specific.

Rather than training a large model from scratch — which requires massive data, compute, and time — we **transfer** those learned representations to our smaller problem.

---

## 2. Why Use It?

| Problem | Transfer Learning Benefit |
|---------|--------------------------|
| Small dataset | Reuses features learned from millions of images |
| Limited compute | Only a small head is trained |
| Faster convergence | Weights start near a useful optimum |
| Better generalization | Pretrained features are robust and diverse |

For TinyML and embedded vision, transfer learning is often the **only practical way** to reach production-quality accuracy.

---

## 3. Two Main Strategies

### 3.1 Feature Extraction (used in this assignment)

- Load the pretrained network **without** its top classification layers (`include_top=False`).
- **Freeze** all convolutional layers (`layer.trainable = False`).
- Attach a **new classification head** (e.g. GlobalAveragePooling + Dense).
- Train **only the head**.

The frozen convolutional base acts as a fixed *feature extractor*, converting each image into a compact feature vector that the small trainable head learns to classify.

### 3.2 Fine-Tuning

- Start as above, then **unfreeze** the top few layers of the base model.
- Continue training with a **very small learning rate** so the pretrained weights are only slightly adjusted.
- Useful when the target domain differs meaningfully from ImageNet.

> In this assignment we use **feature extraction only** — it's simpler and already gets us >95% accuracy.

---

## 4. The "Bottleneck" Layer

In most CNNs, the layer immediately **before** the classification head is the richest, most compact representation of the input — often called the **bottleneck**. In MobileNet-V1 with `include_top=False`, this is the output of the final depthwise-separable convolution block. We take this output, average-pool it, and feed it into our new head.

---

## 5. MobileNet-V1 in Brief

MobileNet-V1 is a lightweight CNN designed for mobile and embedded devices. It uses **depthwise-separable convolutions** to drastically reduce parameters and FLOPs compared to standard convolutions, while retaining strong accuracy. This makes it an excellent backbone for TinyML tasks like Visual Wake Words and mask detection.

Input requirement: pixel values normalized to **[-1, 1]** (handled via `mobilenet.preprocess_input`).

---

## 6. Preventing Overfitting

Because the dataset is small, we use two standard techniques:

- **Data augmentation** — random horizontal flips and rotations applied only during training, so each epoch effectively sees a slightly different dataset.
- **Dropout** — a `Dropout(0.2)` layer before the final Dense output randomly zeros activations during training.

---

## 7. Loss and Output

Since this is a **binary classification** problem, we:

- Use a single-unit `Dense` layer with **linear activation** (i.e. logits).
- Pair it with `BinaryCrossentropy(from_logits=True)` — this is numerically more stable than applying sigmoid inside the model.
- At inference time, apply `tf.nn.sigmoid` and threshold at 0.5.

---

## 8. General ML Workflow Recap

1. **Examine and understand the data.**
2. **Build an input pipeline** (batching, shuffling, prefetching).
3. **Compose the model** (base + head).
4. **Train** with an appropriate optimizer and loss.
5. **Evaluate** on a held-out test set and inspect predictions qualitatively.

Transfer learning fits neatly into step 3 — it changes *how* we compose the model, not the overall workflow.

---

