# 🏥 Diabetes Prediction with PyTorch

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![Deep Learning](https://img.shields.io/badge/Type-Deep_Learning-purple)

## 📌 Project Overview
This project demonstrates the implementation of a **Feed-Forward Neural Network (ANN)** using **PyTorch** to classify diabetes health indicators.

Unlike standard binary classification tasks, this model is designed to handle **Multi-Class Classification** (e.g., Non-Diabetic, Pre-Diabetic, Diabetic), requiring a specific network architecture with multiple output neurons.

## 🧠 Neural Network Architecture
The model is built using `torch.nn.Module` with the following structure:
- **Input Layer:** Accepts 8 physiological features (Glucose, BMI, Age, etc.).
- **Hidden Layers:** Two fully connected layers (20 neurons each) with **ReLU** activation functions to capture non-linear patterns.
- **Output Layer:** 3 neurons corresponding to the target classes (Class 0, 1, 2).

## 🛠️ Tech Stack
- **PyTorch:** Core framework for building and training the neural network.
- **Scikit-Learn:** Used for data splitting (`train_test_split`) and standardization (`StandardScaler`).
- **Pandas & NumPy:** Data manipulation and numerical operations.
- **Matplotlib:** Visualization of training loss curves.

## ⚙️ Methodology
1. **Data Preprocessing:** - Feature scaling using `StandardScaler` (Crucial for Neural Network convergence).
   - Tensor conversion (`FloatTensor`, `LongTensor`).
2. **Model Training:**
   - **Loss Function:** `CrossEntropyLoss` (optimized for multi-class classification).
   - **Optimizer:** `Adam` optimizer with a learning rate of 0.01.
   - **Epochs:** Iterative training to minimize loss.
3. **Evaluation:**
   - Accuracy calculation on unseen test data.
   - Comparison of predicted classes vs. actual ground truth.

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install torch pandas sklearn matplotlib
