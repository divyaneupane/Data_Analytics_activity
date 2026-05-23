import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

 
# 1. LOAD AND PREPARE DATA

print("Loading Iris dataset...")
iris = datasets.load_iris()

# Create DataFrame
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add target column (numeric: 0,1,2)
iris_df['class'] = iris.target

# Rename columns to simpler names (optional)
iris_df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']

# 2. INITIAL DATA INSPECTION
print("\nFirst 5 rows:")
print(iris_df.head())

print("\nMissing values per column:")
print(iris_df.isnull().sum())

print("\nMean of each feature:")
print(iris_df.iloc[:, :4].mean())

# Remove any completely empty rows (none in Iris dataset, but safe)
before = len(iris_df)
iris_df.dropna(how='all', inplace=True)
if len(iris_df) < before:
    print(f"Removed {before - len(iris_df)} all‑NA rows.")

# 3. SCATTER PLOTS (VISUALIZATION)
# Add a temporary species name column for legend
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
iris_df['species_name'] = iris_df['class'].map(species_map)

# Scatter plot 1: Sepal length vs Sepal width
plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris_df, x='sepal_len', y='sepal_wid', hue='species_name')
plt.title('Sepal Scatter Plot')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.savefig('sepal_scatter_plot.png')
plt.show()

# Scatter plot 2: Petal length vs Petal width
plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris_df, x='petal_len', y='petal_wid', hue='species_name')
plt.title('Petal Scatter Plot')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.savefig('petal_scatter_plot.png')
plt.show()

# Remove temporary column
iris_df.drop('species_name', axis=1, inplace=True)

# 
# 4. TRAIN‑TEST SPLIT
X = iris_df[['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']]
y = iris_df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

 
# 5. TRAIN SVM WITH LINEAR KERNEL

print("\nTraining SVM (linear kernel)...")
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)


# 6. EVALUATION
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

target_names = ['setosa', 'versicolor', 'virginica']
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (raw counts):")
print(cm)

# 7. PLOT CONFUSION MATRIX
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')
plt.show()


print("Saved files:")
print("  sepal_scatter_plot.png")
print("  petal_scatter_plot.png")
print("  confusion_matrix.png")