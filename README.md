# 🩺 Diabetes Medical Classification (PyTorch ANN Implementation)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 📌 Project Overview
This project develops an **Artificial Neural Network (ANN)** to classify patients into three medical categories: **Normal**, **Prediabetes**, and **Diabetes**. 

The goal is to demonstrate how deep learning can be applied to clinical tabular data to assist in early medical diagnosis. The model processes various physiological metrics (cholesterol, glucose, BMI, etc.) to predict diabetic status with high precision.

## 📂 Dataset
The dataset used in this project is sourced from **Kaggle**.
- **Source:** [Diabetes Dataset (Kaggle)](https://www.kaggle.com/datasets/imtkaggleteam/diabetes)
- **Description:** Patient clinical records including glycosylated hemoglobin levels, BMI, and blood pressure.
- **Target Variable:** `glyhb_cat` (0: Normal, 1: Prediabetes, 2: Diabetes).

## 🧪 Deep Learning Workflow
The notebook follows a rigorous data science pipeline to ensure model reliability:

1. **Exploratory Data Analysis (EDA):** Detailed visualization of clinical variables using histograms to detect distribution patterns.
2. **Missing Value Management:**
    * **Mean Imputation:** Applied to normally distributed features like blood pressure and cholesterol.
    * **Median Imputation:** Applied to skewed features to maintain robust statistical integrity.
3. **Advanced Preprocessing:**
    * **Anti-Leakage Design:** The `glyhb` column is dropped from the features ($X$) because the target ($y$) is derived from it.
    * **Robust Scaling:** Utilization of `RobustScaler` to handle outliers effectively.
    * **Label Encoding:** Converting text labels into integers (0, 1, 2) for PyTorch compatibility.
4. **ANN Architecture:**
    * **Input Layer:** 14 nodes.
    * **Hidden Layers:** Two dense layers (12 & 24 neurons) using **ReLU** activation.
    * **Output Layer:** 3 nodes representing class probabilities.



## 📊 Evaluation Results
The model is trained for **500 epochs** using the **Adam Optimizer** and **Cross-Entropy Loss**. The evaluation phase includes:
* **Loss Curve Plotting:** Tracking the convergence of the model.
* **Accuracy Calculation:** Performance metrics calculated over a 20% dedicated test set.

## 🚀 Getting Started

### 1. Installation
Clone this repository and install the required Python libraries:

```bash
# Clone the repository
git clone [https://github.com/your-username/diabetes-ann-pytorch.git](https://github.com/your-username/diabetes-ann-pytorch.git)

# Navigate to the project folder
cd diabetes-ann-pytorch

# Install dependencies
pip install torch pandas matplotlib scikit-learn notebook
