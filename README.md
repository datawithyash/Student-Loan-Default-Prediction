
# 🎯 Student Loan Default Prediction

This project uses machine learning to predict whether a student will default on a loan based on financial and educational features.

### 🔍 Features Used:
- Income
- Credit Score
- Loan Amount
- Education Level (encoded)

### 🛠️ Technologies:
- Python
- pandas, scikit-learn, matplotlib, seaborn
- Random Forest Classifier

### ⚙️ What it Does:
- Encodes education level
- Splits data into training and test sets
- Trains a Random Forest model
- Predicts loan default
- Displays accuracy and confusion matrix

### ✅ Sample Output:
=======
# Student-Loan-Default-Prediction
A machine learning project to predict whether a student will default on their loan — helping financial platforms reduce risk and optimize lending decisions.

## 💼 Problem Statement
In the education financing space, student loan defaults create massive risk for lenders. Early identification of high-risk borrowers can enable platforms to personalize loan terms, offer timely interventions, and prevent revenue loss.
This project aims to build a predictive model using historical student data to classify whether a student is likely to default.

## 🧠 Objectives
- Predict loan default using student profile data (GPA, loan amount, income, etc.)
- Understand key drivers behind defaults through EDA
- Provide actionable insights for product and risk teams
- 
## 📊 Dataset
- Source: [Kaggle Dataset - Student Loan Default Prediction](https://www.kaggle.com/datasets/itssuru/student-loan-default-prediction)
- Includes features such as:
  - GPA
  - Loan Amount
  - Family Income
  - Interest Rate
  - Repayment Schedule
  - Employment and Marital Status

## 🔍 Methodology
- **Exploratory Data Analysis (EDA)**
- **Data Cleaning & Encoding**
- **Modeling**:
  - Logistic Regression
  - Random Forest
  - XGBoost (optional)
- **Evaluation**:
  - Accuracy
  - Confusion Matrix
  - ROC-AUC

## 🛠 Tech Stack
- Python
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebooks

## 📈 Business Impact
By integrating such a model into a lending platform, teams can:
- Reduce default rates
- Automate credit scoring
- Personalize loan offers
- Assist in targeted follow-ups

## 🔮 Future Scope
- Add a Streamlit dashboard for real-time predictions
- Use SHAP for feature importance explanation
- Integrate with Tableau for product analytics
- Expand model to include behavioral data from app usage

## 🚀 Run Locally

```bash
git clone https://github.com/datawithyash/student-loan-default-prediction.git
cd student-loan-default-prediction
pip install -r requirements.txt
jupyter notebook
 59656a582e1a5a71a5e28ffb152953eb382c4a11
