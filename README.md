# 👁️ RetinoNet-APTOS

An end-to-end deep learning pipeline built using TensorFlow/Keras to analyze and classify retinal fundus images from the **APTOS 2019 Blindness Detection** challenge. The model handles severe class imbalance using SMOTE and categorizes images into five severity stages of Diabetic Retinopathy (DR).

---

## 📌 Features

* **Data Preprocessing & EDA:** Visualizes class distributions and sample images across all 5 severity levels.
* **Class Balancing via SMOTE:** Applies Synthetic Minority Over-sampling Technique (SMOTE) to flattened image vectors, balancing minority severity classes prior to model training.
* **Custom CNN Architecture:** Built with Keras `Sequential` using multi-layer Convolutional (`Conv2D`), Pooling (`MaxPooling2D`), Dropout, and Dense layers.
* **Detailed Metrics Callback:** Custom Keras callback evaluating Quadratic Weighted Kappa (QWK), Balanced Accuracy, Macro F1, Macro Recall, and per-class metrics live during training.
* **Automated Checkpoints:** Saves the model with the best validation accuracy (`best_model.keras`) and utilizes Early Stopping.

---

## 📊 Severity Classification Scale

| Class Label | Severity Level |
| :---: | :--- |
| **0** | No DR |
| **1** | Mild |
| **2** | Moderate |
| **3** | Severe |
| **4** | Proliferative DR |

---

## 🚀 Quick Start

### 1. Requirements

Ensure you have Python 3.10+ and the required dependencies installed:

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn imbalanced-learn opencv-python

