# ALPAI Project: Semi-Supervised Learning for Post-Earthquake Image Classification

## 🔍 Objective

This project focuses on automating the classification of post-earthquake building images using **semi-supervised learning (SSL)**. The aim is to reduce dependence on expensive and time-consuming manual labeling while maintaining high classification accuracy.

---

## 📂 Dataset

- **Source**: [PEER-Hub ImageNet (phi-Net)](https://apps.peer.berkeley.edu/phi-net/)
- Contains real-world earthquake reconnaissance images.
- Two classification tasks:
  - **Task 1 – Scene Level**
    - `Class 0`: Object-level (component like column or wall)
    - `Class 1`: Pixel-level (surface close-up)
    - `Class 2`: Structural-level (entire building/multiple buildings)
  - **Task 2 – Damage State**
    - `Class 0`: Damaged
    - `Class 1`: Undamaged

---

## 🧠 Methodology

- **Model**: Transfer learning using **VGG16** (ImageNet pretrained)
- **Baseline**: Trained on labeled data only
- **SSL Strategy**: Pseudo-Labeling
  - Generate pseudo-labels for unlabeled images
  - Retrain model using both labeled and pseudo-labeled data
  - Use a weighted loss:  
    \[
    \mathcal{L} = \mathcal{L}_{\text{labeled}} + \alpha(t) \cdot \mathcal{L}_{\text{unlabeled}}
    \]
  - Where α(t) increases over epochs to trust pseudo-labels more gradually



---

## 📎 License

This project is for educational and research purposes.
