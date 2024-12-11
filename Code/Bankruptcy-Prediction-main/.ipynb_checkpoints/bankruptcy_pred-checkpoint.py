import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

df = pd.read_csv("american_bankruptcy.csv")
df = df.drop(columns = ["company_name"])

# 1. DATA PREPROCESSING

# Data Cleaning

missing_values = df.isnull().sum()
# print("\nMissing values per column:\n", missing_values)
duplicates = df.duplicated().sum()
# print("\nDuplicates in the dataset:", duplicates)

# Feature Encoding (Binary Encoding)

# print("\nLabels in the dataset:", df["status_label"].unique()) 
df["status_encoding"] = df["status_label"].map({"alive": 0, "failed": 1})
df["status_encoding"] = df["status_encoding"].astype(int)
df = df.drop(columns = ["status_label"])
# print(df.head(10))

"""

YOUR CODE GOES HERE!!! (Data scaling/standardization comes last.)

"""

# Data Scaling/Standardization

scaler = StandardScaler()

X = df.iloc[:, :-1]
y = df["status_encoding"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 21, test_size = 0.20, shuffle = True)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

train_scaled = pd.concat([pd.DataFrame(X_train_scaled, columns = X.columns), pd.DataFrame(y_train.reset_index(drop = True), columns = ["status_encoding"])], axis = 1)
test_scaled = pd.concat([pd.DataFrame(X_test_scaled, columns = X.columns), pd.DataFrame(y_test.reset_index(drop = True), columns = ["status_encoding"])], axis = 1)
df_scaled = pd.concat([train_scaled, test_scaled], axis = 0)
## print(df_scaled.head(10))