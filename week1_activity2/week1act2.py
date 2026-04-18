import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv(r"D:\Master SE\2nd trimister\Data analytics\week1_activity2\Housing.csv")

# Check first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Check data types
print("\nData types:")
print(df.dtypes)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

# Check for duplicates
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Convert binary columns to numeric
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})  
print("\nData types after conversion:")
print(df.dtypes)

#  VISUALIZATIONS

# 1. Bar chart: Average price by number of bedrooms
plt.figure(figsize=(10, 6))
avg_price_bedrooms = df.groupby('bedrooms')['price'].mean()
plt.bar(avg_price_bedrooms.index, avg_price_bedrooms.values, color='skyblue')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Average Price')
plt.title('Average House Price by Number of Bedrooms')
plt.xticks(avg_price_bedrooms.index)
plt.grid(True, alpha=0.3)
plt.savefig('price_by_bedrooms.png')
plt.show()

# 2. Bar chart: Average price by number of bathrooms
plt.figure(figsize=(10, 6))
avg_price_bathrooms = df.groupby('bathrooms')['price'].mean()
plt.bar(avg_price_bathrooms.index, avg_price_bathrooms.values, color='lightgreen')
plt.xlabel('Number of Bathrooms')
plt.ylabel('Average Price')
plt.title('Average House Price by Number of Bathrooms')
plt.xticks(avg_price_bathrooms.index)
plt.grid(True, alpha=0.3)
plt.savefig('price_by_bathrooms.png')
plt.show()

# 3. Scatter plot: Bedrooms vs Price
plt.figure(figsize=(10, 6))
plt.scatter(df['bedrooms'], df['price'], alpha=0.5, color='blue')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price')
plt.title('House Price vs Number of Bedrooms')
plt.grid(True, alpha=0.3)
plt.savefig('scatter_bedrooms_price.png')
plt.show()

# 4. Scatter plot: Bathrooms vs Price
plt.figure(figsize=(10, 6))
plt.scatter(df['bathrooms'], df['price'], alpha=0.5, color='red')
plt.xlabel('Number of Bathrooms')
plt.ylabel('Price')
plt.title('House Price vs Number of Bathrooms')
plt.grid(True, alpha=0.3)
plt.savefig('scatter_bathrooms_price.png')
plt.show()

# 5. Combined: Total Rooms vs Price
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
plt.figure(figsize=(10, 6))
plt.scatter(df['total_rooms'], df['price'], alpha=0.5, color='purple')
plt.xlabel('Total Rooms (Bedrooms + Bathrooms)')
plt.ylabel('Price')
plt.title('House Price vs Total Rooms')
plt.grid(True, alpha=0.3)
plt.savefig('scatter_totalrooms_price.png')
plt.show()

print("\nAll visualizations saved as PNG files in your current folder!")