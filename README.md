# Brain Tumor Classification using Stacked Ensemble Deep Learning

This project focuses on automated brain tumor classification using deep learning and stacked ensemble techniques on MRI scans. The system accurately classifies MRI images into four clinical categories (Glioma, Meningioma, Pituitary, No Tumor), addressing robustness, calibration, and interpretability challenges in medical image diagnosis.

## Project Description

This project aims to build a clinically rigorous automated brain tumor classification system using a **Cross-Validated Stacked Ensemble** deep learning model with transfer learning. Multiple state-of-the-art pretrained CNN architectures are combined to improve classification accuracy and reliability. To address the critical need for trust in medical AI, the project implements **Probability Calibration**, **Uncertainty Quantification**, and **Explainable AI (XAI)** techniques (Grad-CAM) to interpret model predictions and visualize tumor localization. Four novel clinical contributions extend the baseline: **(A) Uncertainty-Aware Triage** via Temperature Scaling and Test-Time Augmentation, **(B) Prototype-Based Explanation** using embedding similarity retrieval, **(C) Scanner Domain-Shift Robustness** testing against MRI-realistic degradations, and **(D) Deep Uncertainty Characterisation** with per-class entropy profiling.

## Key Steps

### 1) Data Loading and Exploration

* Load the Brain Tumor MRI dataset from Kaggle
* Inspect dataset structure and verify class labels
* Analyze sample MRI scans across four tumor categories
* Identify class distribution to ensure balanced training or handling

### 2) Data Preprocessing

* Resize all MRI scans to `224 × 224 × 3` for model compatibility
* Apply **Data Augmentation** (rotation, shift, zoom) to training data for generalization
* Use a **Clean Validation Set** (no augmentation) for meta-learner training to prevent noise artifacts
* Normalize pixel values using model-specific preprocessing functions

### 3) Model Architecture

Train multiple pretrained CNNs as base learners:

* **ResNet50**: Deep residual network for rich feature extraction
* **EfficientNetB3**: Optimized for efficiency and accuracy
* **DenseNet121**: Feature reuse via dense connectivity
* **InceptionV3**: Multi-scale feature extraction

**Ensemble Strategy:**

* Extract probability vectors independently from each fine-tuned model
* Combine predictions using a **Logistic Regression CV Meta-Learner** (Stacking)
* Optimize ensemble weights based on calibrated likelihoods, not just raw accuracy

### 4) Training Strategy

* Transfer learning using ImageNet-pretrained weights
* **Phase 1**: Feature Extraction (Frozen base layers)
* **Phase 2**: Fine-Tuning (Unfrozen top layers with low learning rate `1e-5`)
* **Optimizer**: Adam
* **Loss Function**: Categorical Cross-Entropy
* **Epochs**: 30 (with Early Stopping for convergence)

### 5) Model Evaluation

Evaluate models using comprehensive clinical metrics:

* **Accuracy**: Overall correctness of diagnosis
* **Sensitivity (Recall)**: Ability to correctly identify tumor cases (crucial for medical screening)
* **Specificity**: Ability to correctly identify non-tumor cases
* **AUC-ROC**: Area Under the Receiver Operating Characteristic curve
* **Confusion Matrices**: Detailed breakdown of diagnostic errors
* **95% Confidence Intervals**: Bootstrapped statistical robustness check

### 6) Explainable AI, Calibration & Novel Contributions

Ensure clinical reliability through advanced validation and four novelty contributions:

* **Grad-CAM**: Visualizing tumor regions to verify model focus
* **Calibration Curves**: Reliability diagrams to check if "90% confidence" means "90% probability"
* **Brier Score**: Metric for the accuracy of probabilistic predictions
* **Entropy Analysis**: Quantifying model uncertainty/confusion
* **Novelty A — Uncertainty-Aware Clinical Triage**: Temperature Scaling (fitted on a held-out calibration split) combined with Test-Time Augmentation (TTA, K=10) to sharpen probability estimates; a confidence threshold automatically routes high-confidence cases to auto-approval and flags low-confidence cases for radiologist review
* **Novelty B — Prototype-Based Explanation**: Nearest-neighbour retrieval in the CNN embedding space (cosine similarity) surfaces the most visually similar confirmed training cases for each query MRI, providing a human-interpretable "this scan resembles these known cases" explanation alongside Grad-CAM
* **Novelty C — Scanner Domain-Shift Robustness**: Systematic stress-testing of the ensemble against three MRI-realistic degradations — Rician noise, k-space undersampling, and bias-field corruption — each evaluated at None / Mild / Moderate / Severe levels to quantify real-world scanner variability tolerance
* **Novelty D — Uncertainty Characterisation**: Per-class entropy profiling, accuracy-vs-confidence-percentile curves, and an active-learning priority queue of the top-8 most uncertain test cases to guide targeted data collection

## Results

The project demonstrates highly effective brain tumor classification with rigorous clinical validation:

**Key Findings:**

* Stacked Ensemble achieves **>98% accuracy** on the test set
* AUC scores exceed **0.99** across all classes
* Calibration analysis confirms valid, trustworthy probability scores (low Brier score)
* Entropy analysis shows significantly reduced uncertainty compared to single models
* Grad-CAM visualizations successfully localize tumor regions without manual segmentation
* Temperature Scaling + TTA further reduces Expected Calibration Error (ECE) and lifts accuracy on the auto-approved triage subset
* Prototype retrieval provides case-based visual explanations with cosine-similarity scores for each prediction
* Domain-shift robustness testing quantifies accuracy drop under Rician noise, k-space undersampling, and bias-field corruption across four severity levels
* Per-class entropy analysis identifies the hardest and easiest tumor classes and shows accuracy gains when retaining only the most confident predictions

## Dataset

**Source**: [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

**Classes**: Glioma, Meningioma, Pituitary, No Tumor

**Size**: ~7,000 MRI images

## Dependencies

The project requires the following Python libraries:

```bash
numpy
pandas
tensorflow
scikit-learn
opencv-python
matplotlib
seaborn
scipy

```

Install the dependencies using:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn scipy opencv-python

```

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/sakshammgarg/BrainTumor_Classification.git
cd BrainTumor_Classification

```

2. **Download the dataset**
* Visit [Kaggle Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
* Download and extract the dataset
* Organize images into the standard directory structure:
```
/input
  /brain-tumor-mri-dataset
    /Training
    /Testing
```

## Usage

Run the notebook to explore the complete analysis and training pipeline.

The notebook will:

1. Load and preprocess the Brain MRI dataset
2. Train individual CNN models (ResNet50, EfficientNetB3, DenseNet121, InceptionV3)
3. Train the **Cross-Validated Stacked Ensemble** meta-learner
4. Display comprehensive evaluation metrics (Accuracy, Sensitivity, Specificity)
5. Generate **Calibration Curves** and **Uncertainty Histograms**
6. Visualize tumor localization using **Grad-CAM** heatmaps
7. Run **Novelty A**: Temperature Scaling + TTA triage pipeline
8. Run **Novelty B**: Prototype-Based Explanation via embedding retrieval
9. Run **Novelty C**: Scanner Domain-Shift Robustness evaluation
10. Run **Novelty D**: Per-class uncertainty characterisation and active-learning queue

## Model Selection Guide

**For Clinical Research:**

* **Best Accuracy & Reliability**: Stacked Ensemble (All models combined)
* **Fast Inference**: EfficientNetB3 (Good balance of speed/accuracy)
* **Feature Richness**: DenseNet121 (Good for small datasets)
* **Maximum Interpretability**: Any model with Grad-CAM applied

## Advanced Features

This comprehensive implementation includes:

* **Stacked Ensemble Learning**: Combining multiple architectures for superior performance
* **Meta-Learning**: Using Logistic Regression CV to learn optimal model weights
* **Robust Validation**: Training meta-learner on clean data to prevent leakage
* **Clinical Calibration**: Ensuring probability scores are mathematically valid
* **Uncertainty Quantification**: Measuring prediction entropy
* **Statistical Rigor**: Bootstrapped Confidence Intervals for accuracy
* **Explainable AI (XAI)**: Grad-CAM for visual tumor verification
* **Uncertainty-Aware Triage (Novelty A)**: Temperature Scaling + TTA with automated confidence-based case routing
* **Prototype-Based Explanation (Novelty B)**: Embedding-space nearest-neighbour retrieval for case-based interpretability
* **Domain-Shift Robustness Testing (Novelty C)**: Stress-testing under Rician noise, k-space undersampling, and bias-field corruption
* **Deep Uncertainty Characterisation (Novelty D)**: Per-class entropy profiling and active-learning priority queue

## Applications

Practical use cases for this brain tumor classification system:

* **Computer-Aided Diagnosis (CAD)** systems for radiologists
* **Triage Systems** to prioritize urgent scans in busy hospitals
* **Second Opinion** tools to reduce diagnostic errors
* **Medical Education** for training students on tumor types
* **Remote Diagnostics** for telemedicine in underserved areas

## Future Improvements

Potential enhancements for even better results:

1. **3D MRI Analysis**: Utilizing volumetric data instead of 2D slices
2. **Tumor Segmentation**: Predicting exact tumor boundaries (masks) alongside classification
3. **Multi-Modal Fusion**: Combining MRI with CT or clinical patient data
4. **Active Learning**: Continuously improving the model with radiologist feedback
5. **Edge Deployment**: Optimizing models for deployment on portable medical devices
6. **External Validation**: Testing on multi-center datasets to verify generalization

---

This project demonstrates how **Stacked Ensemble Learning**, **Probability Calibration**, **Explainable AI**, and four novel clinical contributions (Uncertainty-Aware Triage, Prototype-Based Explanation, Scanner Robustness Testing, and Uncertainty Characterisation) can be combined to build accurate, reliable, and trustworthy medical diagnostic systems suitable for clinical research and application.
