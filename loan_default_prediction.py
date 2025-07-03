# Load real dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("train.csv")

# Select relevant columns
df = df[["ApplicantIncome", "LoanAmount", "Education", "Credit_History", "Loan_Status"]]

# Drop rows with missing values
df.dropna(inplace=True)

# Encode categorical variables
le = LabelEncoder()
df["Education"] = le.fit_transform(df["Education"])
df["Loan_Status"] = df["Loan_Status"].map({"Y": 0, "N": 1})  # 1 = Default, 0 = Paid

# Features and label
X = df[["ApplicantIncome", "LoanAmount", "Education", "Credit_History"]]
y = df["Loan_Status"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# Visualize the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
