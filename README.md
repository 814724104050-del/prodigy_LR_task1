# prodigy_LR_task1
# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("train.csv")   # Kaggle house price dataset

# Select important columns
df = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]
df.dropna(inplace=True)

# Features and target
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, pred)
print("Mean Squared Error :", mse)

print("Prediction Sample:")
print(pred[:10])
