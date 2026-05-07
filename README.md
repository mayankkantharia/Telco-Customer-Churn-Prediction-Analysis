# 📉 Telco Customer Churn Prediction - End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

Predicting customer churn for a telecom provider using a complete ML pipeline: EDA, feature selection, six classifiers, hyperparameter tuning, and a neural network. Best ROC-AUC: **0.836**.

---

## Table of Contents
- [Problem](#problem)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Business Recommendations](#business-recommendations)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Data Source](#data-source)

---

## Problem

Customer churn is a key revenue risk for telecoms. This project identifies at-risk customers using demographics, service usage, and billing data so that targeted retention actions can be taken.

---

## Dataset

- **Source:** [IBM Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** Original 7,043 rows; this work uses a stratified sample of **5,000** rows
- **Features:** 21 (demographics, contract type, payment method, services, billing)
- **Class balance:** ~73% non-churn / 27% churn

---

## Methodology

### Phase 1 - EDA & Data Preparation
- Removed irrelevant columns (e.g., `customerID`) and handled missing `TotalCharges`
- Standardised column names and categorical labels
- Visualised distributions, correlations, and relationships (histograms, boxplots, violin plots)
- Stratified sampling to keep class proportions in the 5,000-record sample

Key EDA findings:
- Month-to-month customers churn more frequently
- Electronic check users show higher churn than auto-pay users
- Higher monthly charges and shorter tenure are strong churn indicators
- Senior citizens and very new customers are higher-risk segments

### Phase 2 - Predictive Modelling

- **Split:** 70/30 stratified train/test
- **Validation:** 5-fold stratified cross-validation
- **Feature selection:** ANOVA F-value ranking
- **Preprocessing:** `StandardScaler` for numerics; One-Hot Encoding for categoricals
- **Dropped:** `Gender`, `PhoneService` (low predictive signal per ANOVA)

Models trained and tuning:
- Logistic Regression - GridSearchCV (C, solver)
- K-Nearest Neighbours - GridSearchCV (k, metric)
- Naive Bayes - var_smoothing
- Decision Tree - GridSearchCV (max_depth, min_samples)
- Random Forest - GridSearchCV (n_estimators, max_depth)
- Neural Network - manual tuning (lr, dropout, batch size, units)

Neural network (summary): 32 → 16 hidden units, ReLU, Sigmoid output, Adam (lr=0.001), dropout 0.10, batch size 64

---

## Results

Selected metrics:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7934 | 0.7855 | 0.7934 | 0.7882 | **0.8359** |
| Random Forest | 0.7953 | 0.7832 | 0.7953 | 0.7846 | 0.8264 |
| Naive Bayes | 0.7829 | 0.7676 | 0.7829 | 0.7684 | 0.8233 |
| Neural Network | 0.7791 | 0.7685 | 0.7791 | 0.7718 | 0.8250 |
| KNN | 0.7787 | 0.7725 | 0.7787 | 0.7750 | 0.8188 |
| Decision Tree | 0.7796 | 0.7624 | 0.7796 | 0.7604 | 0.7927 |

Logistic Regression achieved the best ROC-AUC (0.836), showing strong discriminative power after proper encoding.

Top predictors: Contract type · Monthly charges · Tenure · Payment method · Tech support

---

## Business Recommendations

- Offer incentives to move month-to-month customers to longer contracts
- Encourage electronic-check customers to switch to automatic billing
- Provide onboarding and loyalty incentives for high-bill, short-tenure customers
- Prioritise retention outreach for senior citizens and new customers

---

## Project Structure

```
Telco-Customer-Churn-Prediction-Analysis/
├── Telco Customer Churn Prediction Analysis.ipynb    # Full pipeline notebook (final pipeline)
├── Phase1.ipynb                                      # Phase 1: EDA & data-prep notebook
├── Phase1.csv                                        # Sampled dataset used in Phase 1
├── Phase2.csv                                        # Dataset / split used in Phase 2 modelling
├── README.md                                         # This file
```

---

## How to Run (quick)

1. Clone the repo

```bash
git clone https://github.com/tanishq19/telco-customer-churn.git
cd telco-customer-churn
```

2. Download the dataset from Kaggle and place the CSV in the project root:
   https://www.kaggle.com/datasets/blastchar/telco-customer-churn

3. Install dependencies

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn jupyter
```

4. Launch the notebook

```bash
jupyter notebook "Telco Customer Churn Prediction Analysis.ipynb"
```

---

## Data Source

Blastchar. (2018). *Telco Customer Churn* [Data set]. Kaggle.  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

