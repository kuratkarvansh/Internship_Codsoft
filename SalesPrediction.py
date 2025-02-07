#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
file_path = "advertising.csv"
df = pd.read_csv(file_path)

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(df['TV'], bins=20, kde=True)
plt.title("TV Advertising Budget Distribution")

plt.subplot(1, 3, 2)
sns.histplot(df['Radio'], bins=20, kde=True)
plt.title("Radio Advertising Budget Distribution")

plt.subplot(1, 3, 3)
sns.histplot(df['Newspaper'], bins=20, kde=True)
plt.title("Newspaper Advertising Budget Distribution")
plt.show()

# Scatter plots
sns.pairplot(df, diag_kind='kde')
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Split features and target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2 Score: {r2}')

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# Residual plot
plt.figure(figsize=(8, 6))
sns.residplot(x=y_test, y=y_pred, lowess=True, line_kws={"color": "red"})
plt.xlabel("Actual Sales")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Feature importance
plt.figure(figsize=(8, 6))
sns.barplot(x=X.columns, y=model.coef_)
plt.xlabel("Features")
plt.ylabel("Coefficients")
plt.title("Feature Importance")
plt.show()


# In[ ]:





# In[ ]:




