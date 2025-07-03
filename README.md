# 🎯 Student Loan Default Prediction

This project uses machine learning to predict whether a student will default on a loan using financial and demographic data. It is powered by a real-world dataset from Kaggle and demonstrates the full pipeline of a supervised classification problem — from preprocessing and feature encoding to model training and evaluation.

---

### 📂 Dataset Source:

**Kaggle:** [Loan Prediction Dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication)

The dataset includes:

- `ApplicantIncome`
- `LoanAmount`
- `Education` (Graduate / Not Graduate)
- `Credit_History` (1.0 = has credit history, 0.0 = none)
- `Loan_Status` (`Y` = loan approved, `N` = not approved)

We interpret:
- `Loan_Status = N` → **Default** (`1`)
- `Loan_Status = Y` → **No Default** (`0`)

---

### 🧠 Problem Statement:

> Predict whether a loan applicant is likely to default based on their income, loan amount, credit history, and education level.

---

### 🛠️ Technologies Used:

- Python
- pandas, scikit-learn
- Random Forest Classifier
- matplotlib, seaborn (for visualization)

---

### ⚙️ Machine Learning Pipeline:

1. **Data Cleaning**: Handled missing values using `.dropna()`  
2. **Feature Engineering**: Encoded `Education` and `Loan_Status` using label encoders & mapping  
3. **Train-Test Split**: 70/30 using `train_test_split()`  
4. **Model**: Random Forest Classifier  
5. **Evaluation**:
    - Accuracy score
    - Confusion matrix
    - Heatmap visualization

---

### ✅ Sample Output:

```python
Model Accuracy: 0.84
Confusion Matrix:
[[86  1]
 [15 23]]
