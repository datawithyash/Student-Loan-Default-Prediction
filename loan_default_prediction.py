import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset
data = {
    'income': [30000, 40000, 25000, 50000, 28000, 60000, 70000, 32000, 45000, 52000],
    'credit_score': [650, 700, 580, 720, 600, 750, 800, 640, 670, 710],
    'loan_amount': [10000, 15000, 8000, 12000, 7000, 20000, 25000, 9000, 11000, 18000],
    'education_level': ['graduate', 'postgraduate', 'undergraduate', 'graduate', 'undergraduate',
                        'postgraduate', 'graduate', 'graduate', 'postgraduate', 'undergraduate'],
    'default': [0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
}
df = pd.DataFrame(data)

# Encode categorical variable
le = LabelEncoder()
df['education_level_encoded'] = le.fit_transform(df['education_level'])

# Define features and label
X = df[['income', 'credit_score', 'loan_amount', 'education_level_encoded']]
y = df['default']

# Split the data
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

# Optional: Visualize confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
