import math
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
# from google.colab import drive

df = pd.read_csv('american_bankruptcy.csv')

df = df.drop(columns = ['company_name'])
df = df.drop(columns = ['year'])

print(df.head(10))


# Data Cleaning

missing_values = df.isnull().sum()
print('\nMissing values per column:\n', missing_values)

duplicates = df.duplicated().sum()
print('\nDuplicates in the dataset:', duplicates)


# Feature Encoding

print('\nLabels in the dataset:', df['status_label'].unique())

df['status_encoding'] = df['status_label'].map({'alive': 0, 'failed': 1})
df['status_encoding'] = df['status_encoding'].astype(int)
df = df.drop(columns = ['status_label'])
print(df.head(10))


# Outlier Detection

features_to_plot = df.columns[df.columns != 'status_encoding']

for feature in features_to_plot:
    Q1 = df[feature].quantile(0.25)
    Q2 = df[feature].quantile(0.50)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
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

scaler = StandardScaler()

X = df.iloc[:, :-1]
y = df['status_encoding']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=21, test_size=0.20, shuffle=True)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

train_scaled = pd.concat([pd.DataFrame(X_train_scaled, columns=X.columns), pd.DataFrame(y_train.reset_index(drop=True), columns =['status_encoding'])], axis=1)
test_scaled = pd.concat([pd.DataFrame(X_test_scaled, columns=X.columns), pd.DataFrame(y_test.reset_index(drop=True), columns =['status_encoding'])], axis=1)
df_scaled = pd.concat([train_scaled, test_scaled], axis=0)
print(df_scaled.head(10))

print('/n')

diagram_columns = 3
attribute_columns = df_scaled.shape[1] - 1
diagram_rows = math.ceil(attribute_columns / diagram_columns)

row_index = 0
column_index = 0
fig, axes = plt.subplots(diagram_rows, diagram_columns, figsize=(18, diagram_rows * 3))
for column in df_scaled.columns[:-1]:
    sns.boxplot(x='status_encoding', y=column, data=df_scaled, ax=axes[row_index, column_index])
    axes[row_index, column_index].set_title(f'Box Plot of {column} (Z-Score Standardized)')
    axes[row_index, column_index].set_xlabel('Bankruptcy Status')
    axes[row_index, column_index].set_ylabel(f'{column} (Z-Score Standardized)')
    column_index += 1
    if column_index == diagram_columns:
        row_index += 1
        column_index = 0
plt.tight_layout()
plt.show()

print('\n')

row_index = 0
column_index = 0
fig, axes = plt.subplots(diagram_rows, diagram_columns, figsize=(18, diagram_rows * 3))
for column in df_scaled.columns[:-1]:
    sns.violinplot(x='status_encoding', y=column, data=df_scaled, ax=axes[row_index, column_index])
    axes[row_index, column_index].set_title(f'Violin Plot of {column} (Z-Score Standardized)')
    axes[row_index, column_index].set_xlabel('Bankruptcy Status')
    axes[row_index, column_index].set_ylabel(f'{column} (Z-Score Standardized)')
    column_index += 1
    if column_index == diagram_columns:
        row_index += 1
        column_index = 0
plt.tight_layout()
plt.show()


# Applying the Logistic Regression Model

param_grid = {
    'penalty': ['l1', 'l2'], # L1 and L2 regularization 
    'C': [0.01, 0.1, 1, 10, 100], # C is 1 / lambda (i.e. the inverse of the regularization parameter)
    'max_iter': [100, 200, 300, 400] # The maximum number of iterations, which is similar, but not analogous to epochs
}

model = LogisticRegression(solver='liblinear')

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

grid_result = grid.fit(X_train_scaled, y_train)

best_params = grid_result.best_params_
best_score = grid_result.best_score_

print('The Best Model\'s Hyper-Parameters:', best_params)
print('Accuracy of the Best Model: {:.2f}%'.format(best_score * 100))

best_model = LogisticRegression(penalty=best_params['penalty'], C=best_params['C'], solver='liblinear', max_iter=best_params['max_iter'])

history = best_model.fit(X_train_scaled, y_train) # Can use history for plotting useful graphs

y_pred = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print('Accuracy of the Best Model on the Testing Data: {:.2f}%'.format(accuracy * 100))

y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

threshold = 0.5

y_pred_adjusted = (y_pred_proba >= threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred_adjusted)

print(f'Accuracy of the Best Model With the Threshold {threshold}: {accuracy * 100:.2f}%')

cm = confusion_matrix(y_test, y_pred_adjusted)

# Visualize correlation matrix using heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='coolwarm', fmt='d')
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.show()