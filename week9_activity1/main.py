import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sklearn


# Load dataset
df = pd.read_excel(r"D:\Master SE\2nd trimister\Data analytics\week9_activity1\Fitness_App_User_Data.xlsx")

# Display first few rows
print("Dataset preview:")
print(df.head())


#available columns
print("\nAvailable columns:\n", df.columns)

#check for total users
total_users = df['User_ID'].nunique()
print(f"\nTotal unique users: {total_users}")

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

#check for duplicates
print("\nDuplicate rows:", df.duplicated().sum())

#check for negative values in numeric columns
print("\nNegative values in numeric columns:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:    
    neg_count = (df[col] < 0).sum()
    if neg_count > 0:
        print(f"{col}: {neg_count} negative values")                                    
    else:
        print(f"{col}: No negative values")

# #remove negative values in numeric columns
for col in numeric_cols:
    if (df[col]<0).sum() > 0:
        median_value = df[col][df[col] >= 0].median()
        df.loc[df[col] < 0, col] = median_value
        print(f"Replaced negative values in {col} with median: {median_value}")
    else:
        print(f"No negative values to replace in {col}")
        

#task 2
#selected features
selected_features = ['Age', 'workouts_per_week', ]