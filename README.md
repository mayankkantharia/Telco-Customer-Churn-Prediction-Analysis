# 📉 Telco Customer Churn Prediction — End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

> Predicting customer churn for a telecom provider using a full ML pipeline — EDA, feature selection, 6 classifiers, hyperparameter tuning, and a neural network — achieving a best ROC-AUC of **0.836**.

---

## 📌 Problem Statement

Customer churn is a critical business problem in the telecom industry. Identifying at-risk customers before they leave allows companies to intervene with targeted retention strategies. This project builds and evaluates a suite of machine learning models to predict churn using customer demographics, service usage, and billing data.

---

## 📂 Dataset

- **Source:** [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 records → stratified sample of **5,000 records**
- **Features:** 21 (demographics, contract type, payment method, services, billing)
- **Class distribution:** 73% non-churn / 27% churn (imbalanced)

---

## 🔬 Methodology

### Phase 1 — EDA & Data Preparation
- Removed irrelevant columns (e.g. `customerID`), handled missing values in `TotalCharges`
- Standardised column names, data types, and categorical labels
- Conducted univariate, bivariate, and multivariate analysis (histograms, boxplots, violin plots, correlation matrices)
- Stratified random sampling to maintain class balance at 5,000 records

**Key EDA findings:**
- Month-to-month contract customers churn significantly more
- Electronic check users have higher churn rates than auto-pay users
- Higher monthly charges and shorter tenure are strong churn indicators
- Senior citizens and new customers are higher-risk segments

### Phase 2 — Predictive Modelling

**Train/Test Split:** 70/30 stratified split  
**Validation:** 5-fold stratified cross-validation  
**Feature Selection:** ANOVA F-value ranking to identify top predictors  
**Preprocessing:** StandardScaler for numeric features, One-Hot Encoding for categoricals  
**Dropped features:** `Gender`, `PhoneService` (low ANOVA significance)

**Models trained & tuned:**

| Model | Tuning Method |
|---|---|
| Logistic Regression | GridSearchCV (C, solver) |
| K-Nearest Neighbours | GridSearchCV (k, distance metric) |
| Naive Bayes | var_smoothing |
| Decision Tree | GridSearchCV (max_depth, min_samples) |
| Random Forest | GridSearchCV (n_estimators, max_depth) |
| Neural Network | Manual tuning (lr, dropout, batch size, units) |

**Neural Network Architecture:**
- Hidden layers: 32 → 16 units
- Activation: ReLU + Sigmoid output
- Optimizer: Adam (lr = 0.001)
- Dropout: 0.10
- Batch size: 64

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.7934 | 0.7855 | 0.7934 | 0.7882 | **0.8359** ✅ |
| **Random Forest** | 0.7953 | 0.7832 | 0.7953 | 0.7846 | 0.8264 |
| Naive Bayes | 0.7829 | 0.7676 | 0.7829 | 0.7684 | 0.8233 |
| Neural Network | 0.7791 | 0.7685 | 0.7791 | 0.7718 | 0.8250 |
| KNN | 0.7787 | 0.7725 | 0.7787 | 0.7750 | 0.8188 |
| Decision Tree | 0.7796 | 0.7624 | 0.7796 | 0.7604 | 0.7927 |

> Logistic Regression achieved the best ROC-AUC (0.836), demonstrating that once categorical features are properly one-hot encoded, a linear decision boundary is highly effective for this problem.

**Top churn predictors:** Contract type · Monthly charges · Tenure · Payment method · Tech support

---

## 💡 Business Recommendations

- **Contract incentives** — Offer discounts to move month-to-month customers onto 1–2 year contracts
- **Payment outreach** — Encourage electronic check users to switch to automatic billing
- **Loyalty pricing** — Target high-bill, short-tenure customers with onboarding support and loyalty discounts
- **Segment focus** — Prioritise senior citizens and new customers in retention campaigns

---

## 🗂️ Project Structure

```
telco-customer-churn/
│
├── Telco Customer Churn Prediction Analysis.ipynb    # Full pipeline notebook (final pipeline)
├── Phase1.ipynb                                      # Phase 1: EDA & data-prep notebook
├── Phase1.csv                                        # Sampled dataset used in Phase 1
├── Phase2.csv                                        # Dataset / split used in Phase 2 modelling
├── Telco.zip                                         # Original dataset archive (if provided)
├── README.md                                         # Project overview and instructions
```



---

## ▶️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/tanishq19/telco-customer-churn.git
cd telco-customer-churn

# 2. Download the dataset from Kaggle:
#    https://www.kaggle.com/datasets/blastchar/telco-customer-churn
#    Place the CSV in the project root directory

# 3. Install dependencies
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn jupyter

# 4. Launch notebook
jupyter notebook Telco_Customer_Churn_Prediction_Analysis.ipynb
```

---

## 🔗 Data Source

Blastchar. (2018). *Telco Customer Churn* [Data set]. Kaggle.  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

