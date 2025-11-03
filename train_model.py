"""
Script to train the logistic regression model for purchase prediction
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load data
df = pd.read_csv('UserData.csv')  

print("Dataset loaded successfully!")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# Prepare features
# Encode Gender
le = LabelEncoder()
df['Gender_Encoded'] = le.fit_transform(df['Gender'])

# Features and target
X = df[['Gender_Encoded', 'Age', 'EstimatedSalary']].values
y = df['Purchased'].values

print(f"\nClass distribution:")
print(f"Not Purchased (0): {sum(y==0)}")
print(f"Purchased (1): {sum(y==1)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n=== Model Metrics ===")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nConfusion Matrix:")
print(conf_matrix)
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, 'modelo_regresion_logistica.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("\n✓ Model saved as 'modelo_regresion_logistica.pkl'")
print("✓ Scaler saved as 'scaler.pkl'")
print("✓ Label encoder saved as 'label_encoder.pkl'")

# Test predictions
print("\n=== Test Predictions ===")
test_cases = [
    ('Male', 25, 30000),
    ('Female', 35, 75000),
    ('Male', 45, 120000)
]

for gender, age, salary in test_cases:
    gender_enc = 1 if gender == 'Male' else 0
    X_test_case = scaler.transform([[gender_enc, age, salary]])
    prob = model.predict_proba(X_test_case)[0]
    prediction = model.predict(X_test_case)[0]
    print(f"{gender}, {age} years, ${salary:,} -> Probability: {prob[1]:.2%} | Purchased: {'Yes' if prediction == 1 else 'No'}")