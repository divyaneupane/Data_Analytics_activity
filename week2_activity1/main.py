import os
import pandas as pd
import glob

# Set path to the folder containing the CSV files
path = os.path.dirname(os.path.abspath(__file__))

# Find all CSV files
files = glob.glob(os.path.join(path, "*.csv"))
print(f"Found {len(files)} CSV files in {path}")

if not files:
    raise FileNotFoundError(f"No CSV files found in {path}")

# Load and combine
df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list, ignore_index=True)

# Task 1
print("\nFIRST 5 ROWS:")
print(df.head())

print("\nCOLUMN NAMES & DATA TYPES:")
print(df.dtypes)

print(f"\nTOTAL ROWS: {df.shape[0]:,}")
print(f"TOTAL COLUMNS: {df.shape[1]}")