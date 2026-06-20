import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

# 1. LOAD AND PREPARE DATA
print("Loading Wine dataset...")
wine=datasets.load_wine()

# Create DataFrame
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Add target column (numeric: 0,1,2)
wine_df['class'] = wine.target

# 2. INITIAL DATA INSPECTION
print("\nFirst 5 rows:")
print(wine_df.head())

print("\nMissing values per column:")
print(wine_df.isnull().sum())

print("\nMean of each feature:")
print(wine_df.iloc[:, :13].mean())


# Remove any completely empty rows (none in Wine dataset, but safe)
before = len(wine_df)   
wine_df.dropna(how='all', inplace=True)
if len(wine_df) < before:
    print(f"Removed {before - len(wine_df)} all‑NA rows.")
