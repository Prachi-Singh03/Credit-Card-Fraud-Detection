# 💳 Credit Card Fraud Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-GradientBoosting-red)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end machine learning pipeline for detecting fraudulent credit card transactions on a highly imbalanced dataset using data preprocessing, SMOTE, feature engineering, and ensemble learning models.

---

## 📌 Overview

Credit card fraud is a rare-event classification problem where fraudulent transactions account for less than **0.2%** of the data. This project focuses on building a robust fraud detection pipeline capable of handling severe class imbalance while maximizing fraud detection performance.

---

## 🚀 Features

- ✅ End-to-end fraud detection pipeline
- ✅ Feature engineering and preprocessing
- ✅ RobustScaler for outlier handling
- ✅ SMOTE for class imbalance
- ✅ Multiple ML model comparison
- ✅ Hyperparameter tuning
- ✅ Cross-validation
- ✅ Performance evaluation using ROC-AUC, Precision, Recall, and F1-score

---

## 🏗️ Project Workflow

```text
Transaction Data
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Robust Scaling
        │
        ▼
SMOTE (Training Only)
        │
        ▼
Machine Learning Models
(Logistic Regression | Random Forest | XGBoost)
        │
        ▼
Performance Evaluation
```

---

## 📂 Project Structure

```text
Credit_Card_Fraud_Detection/
│
├── data/
│   └── creditcard.csv
│
├── fraud_detection.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| ML | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Imbalanced Learning | SMOTE |
| Visualization | Matplotlib, Seaborn |

---

## ⚙️ Installation

```bash
git clone https://github.com/Pooja3706/Credit-Card-Fraud-Detection.git

cd Credit-Card-Fraud-Detection

pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Quick Run

```bash
python fraud_detection.py --quick
```

### Full Training

```bash
python fraud_detection.py
```

---

## 🤖 Models Evaluated

- Logistic Regression
- Random Forest
- XGBoost

---

## 📊 Evaluation Metrics

- ROC-AUC
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 🎯 Results

✔ Built a class-imbalance-aware fraud detection pipeline using **SMOTE** and **RobustScaler**.

✔ Compared Logistic Regression, Random Forest, and XGBoost to identify the best-performing model.

✔ Achieved **ROC-AUC of 0.92** while improving fraud detection recall on highly imbalanced data.

---

## 📸 Sample Outputs

### Confusion Matrix

> Add confusion matrix screenshot here

```
images/confusion_matrix.png
```

### ROC Curve

> Add ROC curve screenshot here

```
images/roc_curve.png
```

### Model Comparison

> Add model comparison chart here

```
images/model_comparison.png
```

---

## 🔮 Future Improvements

- LightGBM and CatBoost comparison
- Deep Learning Autoencoder
- Explainability using SHAP
- Real-time fraud detection API with FastAPI
- Streamlit Dashboard
- Docker Deployment

---

## 👩‍💻 Author

**Pooja Ratna Sai Sri Teku**

- GitHub: https://github.com/Pooja3706
- LinkedIn: https://linkedin.com/in/pooja-teku

---

⭐ If you found this project useful, consider giving it a star.
