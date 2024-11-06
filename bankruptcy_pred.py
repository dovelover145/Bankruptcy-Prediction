
import math
import numpy as np
import pandas as pd
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

df = pd.read_csv("american_bankruptcy.csv")
df = df.drop(columns = ["company_name"])
df = df.drop(columns = ["year"])

# 1. DATA PREPROCESSING

# Data Cleaning

missing_values = df.isnull().sum()
print("\nMissing values per column:\n", missing_values)
duplicates = df.duplicated().sum()
print("\nDuplicates in the dataset:", duplicates)

# Feature Encoding (Binary Encoding)

print("\nLabels in the dataset:", df["status_label"].unique()) 
df["status_encoding"] = df["status_label"].map({"alive": 0, "failed": 1})
df["status_encoding"] = df["status_encoding"].astype(int)
df = df.drop(columns = ["status_label"])
print(df.head(10))

# 2. EDA

features_to_plot = df.columns

for feature in features_to_plot:
  Q1 = df[feature].quantile(0.25)
  Q2 = df[feature].quantile(0.50)
  Q3 = df[feature].quantile(0.75)
  IQR = Q3- Q1
  lowerBound = Q1 - 1.5 * IQR
  upperBound = Q3 + 1.5 * IQR
  df[feature] = df[feature].clip(lower=lowerBound, upper=upperBound)

num_features = len(features_to_plot)
num_cols = 3
num_rows = math.ceil(num_features / num_cols)

plt.figure(figsize=(12, 2 * num_rows))
for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(num_rows, num_cols, i)
    sns.boxplot(x=df[feature])

plt.tight_layout()
plt.show()

# Data Scaling/Standardization

scaler = StandardScaler() # Scaler for Z-Score standardization

X = df.iloc[:, :-1] # Select all of the rows in the dataset, but disregard the last column (i.e. the target column, "status_encoding")
y = df["status_encoding"] # Select the "status_encoding" column (i.e. the target column)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 21, test_size = 0.20, shuffle = True) # Split the dataset

X_train_scaled = scaler.fit_transform(X_train) # Calculate the mean and standard deviation of X_train and perform the Z-Score standardization
X_test_scaled = scaler.transform(X_test) # Apply it on X_test with the previously calculated mean and standard deviation to avoid data leakage

train_scaled = pd.concat([pd.DataFrame(X_train_scaled, columns = X.columns), pd.DataFrame(y_train.reset_index(drop = True), columns = ["status_encoding"])], axis = 1)
test_scaled = pd.concat([pd.DataFrame(X_test_scaled, columns = X.columns), pd.DataFrame(y_test.reset_index(drop = True), columns = ["status_encoding"])], axis = 1)
df_scaled = pd.concat([train_scaled, test_scaled], axis = 0) # Contains our dataset, but scaled using Z-Score standardization
print(df_scaled.head(10))
print("\n\n\n\n")

diagram_columns = 4 # Can be adjusted to make all of the graphs fit perfectly depending on the number of selected attribute columns
attribute_columns = df_scaled.shape[1] - 1 # Remember to disregard the target column, "status_encoding", as it is not one of our attribute columns
diagram_rows = math.ceil(attribute_columns / diagram_columns) # Find the maximum number of rows in our diagram to fit every attribute's plot

row_index = 0
column_index = 0
fig, axes = plt.subplots(diagram_rows, diagram_columns, figsize = (18, diagram_rows * 3))
for column in df_scaled.columns[:-1]:
    sns.boxplot(x = "status_encoding", y = column, data = df_scaled, ax = axes[row_index, column_index])
    axes[row_index, column_index].set_title(f"Box Plot of {column} (Z-Score Standardized)")
    axes[row_index, column_index].set_xlabel("Bankruptcy Status")
    axes[row_index, column_index].set_ylabel(f"{column} (Z-Score Standardized)")
    column_index += 1
    if column_index == diagram_columns:
        row_index += 1
        column_index = 0
plt.tight_layout()
plt.show()

print("\n\n\n\n")

row_index = 0
column_index = 0
fig, axes = plt.subplots(diagram_rows, diagram_columns, figsize = (18, diagram_rows * 3))
for column in df_scaled.columns[:-1]:
    sns.violinplot(x = "status_encoding", y = column, data = df_scaled, ax = axes[row_index, column_index])
    axes[row_index, column_index].set_title(f"Violin Plot of {column} (Z-Score Standardized)")
    axes[row_index, column_index].set_xlabel("Bankruptcy Status")
    axes[row_index, column_index].set_ylabel(f"{column} (Z-Score Standardized)")
    column_index += 1
    if column_index == diagram_columns:
        row_index += 1
        column_index = 0
plt.tight_layout()
plt.show()