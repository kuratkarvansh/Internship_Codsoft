#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Install the necessary libraries
# pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, precision_recall_curve, auc

# Step 1: Load the dataset
df = pd.read_csv('creditcard.csv')

# Check dataset structure
print(df.head())
print(df.info())

# Check class distribution
print("Class distribution:\n", df['Class'].value_counts())

# Step 2: Preprocess the data
# Normalize 'Amount' feature (PCA features are already normalized)
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Separate features and target variable
X = df.drop(columns=['Class', 'Time'])
y = df['Class']

# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 4: Train the models
# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 5: Evaluate the models
def evaluate_model(model, X_test, y_test):
    # Get model predictions
    y_pred = model.predict(X_test)

    # Classification report (Precision, Recall, F1-Score)
    print(f"Classification Report for {model.__class__.__name__}:")
    print(classification_report(y_test, y_pred))

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)
    print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Evaluate Logistic Regression
evaluate_model(log_model, X_test, y_test)

# Evaluate Random Forest Classifier
evaluate_model(rf_model, X_test, y_test)

# Step 6: Plot Precision-Recall curve for both models
def plot_precision_recall_curve(model, X_test, y_test, model_name):
    precision, recall, _ = precision_recall_curve(y_test, model.predict(X_test))
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.2f})')

plt.figure(figsize=(8, 6))
plot_precision_recall_curve(log_model, X_test, y_test, 'Logistic Regression')
plot_precision_recall_curve(rf_model, X_test, y_test, 'Random Forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.show()

# Step 7: Save the best model (Optional)
import joblib

# Save Random Forest model
joblib.dump(rf_model, 'fraud_detection_rf_model.pkl')
print("Model saved as fraud_detection_rf_model.pkl")


# In[ ]:




