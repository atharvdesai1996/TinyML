# Mask Detection using Transfer Learning

This project uses **Transfer Learning** with a **MobileNet-V1** backbone (pre-trained on ImageNet) to classify whether a person is wearing a face mask or not. It is a lightweight Visual Wake Words style task, well suited for TinyML deployment.

---

## 📁 Folder Contents

| File | Description |
|------|-------------|
| `README.md` | Project overview & usage instructions (this file) |
| `Transfer_Learning_Theory.md` | Background theory on transfer learning, feature extraction & fine-tuning |
| `mask_detection_transfer_learning.py` | Full training pipeline with step-by-step explanations |

---

## 🎯 Objective

Build a binary image classifier (`with_mask` vs `without_mask`) by:

1. Reusing the convolutional feature extractor of MobileNet-V1.
2. Freezing the base model so pretrained weights stay intact.
3. Attaching a small custom classification head.
4. Training only the new head on the mask dataset.

---

## 🧩 Dataset

A modified version of the [Kaggle Face Mask Lite Dataset](https://www.kaggle.com/prasoonkottarathil/face-mask-lite-dataset), pre-split into `train/` and `validation/` directories, each with two subfolders:

```
edx_transfer_learning/
├── train/
│   ├── with_mask/
│   └── without_mask/
└── validation/
    ├── with_mask/
    └── without_mask/
```

Images are resized to **96×96** to keep the model small enough for embedded targets.

---

## ⚙️ Workflow

Following the standard ML workflow:

1. **Examine the data** — visualize sample images and class balance.
2. **Build an input pipeline** using `image_dataset_from_directory` + `prefetch`.
3. **Create a test split** by carving 20% of the validation set.
4. **Data augmentation** — random flips & rotations to reduce overfitting.
5. **Compose the model**
   - Load MobileNet-V1 with `include_top=False`, `weights='imagenet'`.
   - Freeze base model (`trainable = False`).
   - Add `GlobalAveragePooling2D` → `Dropout(0.2)` → `Dense(1)` head.
6. **Compile** with Adam (`lr=1e-4`) and `BinaryCrossentropy(from_logits=True)`.
7. **Train** for a small number of epochs (10 is more than enough).
8. **Evaluate** on the held-out test set and visualize predictions.

---

## 🚀 How to Run

Recommended: run in Google Colab (GPU runtime).

```bash
pip install tensorflow tf_keras gdown
python mask_detection_transfer_learning.py
```

Or open the script in Colab and run cell-by-cell.

---

## 📈 Expected Results

- Initial (untrained head) accuracy: **~50%** (random).
- After only **~10 epochs**: **>95% validation accuracy**.
- Test accuracy is typically very close to validation accuracy given the frozen backbone.

This demonstrates the core value of transfer learning — high accuracy with a small dataset and minimal training time.

---

## 🧠 Key Takeaways

- Transfer learning lets you reuse rich visual features learned on ImageNet.
- Freezing the backbone gives fast, stable training on small datasets.
- A tiny classification head (a single `Dense` layer) is often enough.
- MobileNet-V1 at 96×96 input is a great trade-off for TinyML deployment.

See [`Transfer_Learning_Theory.md`](./Transfer_Learning_Theory.md) for the theoretical background.
