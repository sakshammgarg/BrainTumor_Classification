# Robust Brain Tumor Classification with Stacked Ensemble Deep Learning

Brain tumors require accurate and timely diagnosis. This repository contains an advanced, clinically rigorous deep learning framework for classifying Brain MRI scans into four categories: **Glioma, Meningioma, Pituitary, and No Tumor**. 

Unlike standard implementations, this framework addresses critical issues in medical AI such as probability calibration, data leakage, and uncertainty quantification. It utilizes a **Cross-Validated Stacked Ensemble** of four state-of-the-art architectures, achieving superior performance with mathematically valid confidence scores.

## Key Features & Methodological Improvements

This project moves beyond simple "accuracy" metrics to ensure clinical reliability:

* **Stacked Ensemble Architecture:** Combines **ResNet50**, **EfficientNetB3**, **DenseNet121**, and **InceptionV3**.
* **Meta-Learner:** Uses **LogisticRegressionCV (5-Fold)** to learn optimal weights based on probability likelihoods rather than simple averaging.
* **Robust Validation:** Prevents data leakage by training the meta-learner on a dedicated *clean* validation set (no augmentation noise).
* **Clinical Calibration:** Includes **Reliability Diagrams (Calibration Curves)** and **Brier Scores** to ensure predicted probabilities match real-world correctness.
* **Uncertainty Quantification:** Analyzes prediction **Entropy** to detect model confusion.
* **Statistical Rigor:** Calculates **Bootstrapped 95% Confidence Intervals** for test accuracy.
* **Explainable AI (XAI):** Implements robust **Grad-CAM** (Gradient-weighted Class Activation Mapping) to visualize tumor localization.

## Pipeline Overview

1.  **Data Augmentation:** Rotation, shift, and zoom applied *only* to the training set to improve generalization.
2.  **Feature Extraction:** Transfer learning using ImageNet weights with custom classification heads.
3.  **Fine-Tuning:** Unfreezing top layers with a low learning rate (`1e-5`) for domain adaptation.
4.  **Ensemble Stacking:** Extracting probability vectors from a clean validation set to train the meta-learner.
5.  **Evaluation:** Testing on a hold-out test set with comprehensive clinical metrics.

## Results

The model is evaluated on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

| Metric | Performance |
| :--- | :--- |
| **Accuracy** | **> 98%** (Check specific run logs) |
| **AUC Score** | **> 0.99** (Micro-average) |
| **Calibration** | Low Brier Score (High Trustworthiness) |
| **Sensitivity** | High sensitivity across all tumor types |

*(See the `Outputs/` folder for generated ROC Curves, Confusion Matrices, and Grad-CAM visualizations).*

## 🛠️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Robust-Brain-Tumor-Classification-Ensemble.git](https://github.com/sakshammgarg/Brain_Tumor_Classification.git)
    cd Brain_Tumor_Classification
    ```

2.  **Install dependencies:**
    ```bash
    pip install tensorflow numpy pandas matplotlib seaborn scikit-learn scipy opencv-python
    ```

3.  **Dataset Setup:**
    Download the dataset from Kaggle and place it in the directory structure:
    ```
    /input
      /brain-tumor-mri-dataset
        /Training
        /Testing
    ```

4.  **Run the Script:**
    ```bash
    python main.py
    ```

## 🖼️ Explainability (Grad-CAM)

The framework generates heatmaps to verify that the model is looking at the tumor and not background artifacts.

*(You can upload one of your Grad-CAM screenshots here, e.g., `![Grad-CAM](path_to_image.jpg)`)*

## 🤝 Contribution
Contributions are welcome! Please feel free to submit a Pull Request.

## 🔗 Credits
* **Dataset:** [Masoud Nickparvar (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
* **Frameworks:** TensorFlow/Keras, Scikit-Learn.
